from tensorflow.contrib import rnn, legacy_seq2seq
from collections import Counter
import tensorflow as tf
import numpy as np
import itertools
import pickle
import nltk
import time
import json
import sys
import re
import os

from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits
UNKNOWN = '<unk>'


def update_progress(current, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


class DataModel:
    MIN_WORD_COUNT = 1

    def __init__(self, batch_size, seq_length, save_dir, train_percent=0.85, dev_percent=0):
        self.save_dir = save_dir

        print('Pre-processing...')
        start_time = time.time()
        train_data = dev_data = test_data = []
        files = nltk.corpus.gutenberg.fileids()
        for file in files:
            data = nltk.corpus.gutenberg.raw(file)
            f_train_data, f_dev_data, f_test_data = self.split_data(self.clean_text(data), train_percent, dev_percent)
            train_data += self.clean_tokenize(f_train_data)
            dev_data += self.clean_tokenize(f_dev_data)
            test_data += self.clean_tokenize(f_test_data)

        words = itertools.chain.from_iterable(train_data)
        w_counter = Counter(words)
        words = train_data = self.put_unknown(train_data, w_counter)
        dev_data = self.put_unknown(train_data, w_counter)
        test_data = self.put_unknown(train_data, w_counter)

        json_content = json.dumps({
            'train_data': train_data,
            'dev_data': dev_data,
            'test_data': test_data
        }, indent=4)
        with open(self.get_file_name('gutenberg.json'), 'w') as f:
            f.write(json_content)

        self.vocab = sorted(set(words))
        self.vocab_size = len(self.vocab)
        self.train_len = len(words)
        self.word_to_value = dict((word, i) for i, word in enumerate(self.vocab))
        self.word_value_set = np.array([self.word_to_value[word] for word in words])
        self.save_object(self.vocab, self.get_file_name('vocab.pkl'))
        self.save_object(self.word_to_value, self.get_file_name('word_to_value.pkl'))
        np.save(self.get_file_name('word_value_set.npy'), self.word_value_set)
        print('Total Unique Words: %d' % self.vocab_size)
        print('Total Words: %d' % self.train_len)

        self.n_batches = int((len(self.word_value_set) - 1) / (batch_size * seq_length))
        limit = self.n_batches * batch_size * seq_length
        data_x = self.word_value_set[:limit]
        data_y = self.word_value_set[1:limit + 1]
        self.batch_x = np.split(np.reshape(data_x, [batch_size, -1]), self.n_batches, 1)
        self.batch_y = np.split(np.reshape(data_y, [batch_size, -1]), self.n_batches, 1)
        end_time = time.time()
        print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))

    def put_unknown(self, data, w_counter):
        words = itertools.chain.from_iterable(data)
        return [UNKNOWN if w_counter[word] <= self.MIN_WORD_COUNT else word for word in words]

    @staticmethod
    def clean_text(text):
        text = re.sub('([\d]+)|(\'\')|(\s[\']+)|[";*:`~$]+|[-]{2,}|([\s]+-[\s]+)|[()\[\]]|([\n])', ' ', text)
        text = text.lower()
        return nltk.tokenize.sent_tokenize(text)

    @staticmethod
    def clean_tokenize(text):
        sentences = []
        for sent in text:
            words = ['<s>'] + nltk.tokenize.word_tokenize(sent)
            words[-1] = '</s>'
            sentences += [words]
        return sentences

    @staticmethod
    def split_data(sentences, train_percent, dev_percent):
        total_length = len(sentences)
        train_set = int(train_percent * total_length)
        dev_set = int(train_set + dev_percent * total_length)
        return sentences[:train_set], sentences[train_set:dev_set], sentences[dev_set:]

    @staticmethod
    def save_object(obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def get_file_name(self, file_name):
        return self.save_dir + '/' + file_name

    def get_vocab_size(self):
        return self.vocab_size

    def num_batches(self):
        return self.n_batches

    def get_batch(self, b):
        return self.batch_x[b], self.batch_y[b]


class TokenLSTM:
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])

        cells = []
        for _ in range(args.rnn_layers):
            cells.append(rnn.BasicLSTMCell(args.rnn_size))
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        dense_layer_w = tf.get_variable("dense_layer_w", [args.rnn_size, args.vocab_size])
        dense_layer_b = tf.get_variable("dense_layer_b", [args.vocab_size])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(ip, [1]) for ip in inputs]

        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        outputs, self.final_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell)
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        logits = tf.matmul(output, dense_layer_w) + dense_layer_b
        self.probs = tf.nn.softmax(logits)
        self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, args.seq_length])

        self.lr = tf.Variable(0.0, trainable=False)
        loss = sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.targets, [-1]))
        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def get_file_name(self, file_name):
        return self.args.save_dir + '/' + file_name

    def train(self, data_model):
        args = self.args
        init = tf.global_variables_initializer()
        num_batches = data_model.num_batches()
        tf_saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(init)
            for e in range(args.epochs):
                sess.run(tf.assign(self.lr, args.lr * (args.decay ** e)))
                for b in range(num_batches):
                    state = sess.run(self.initial_state)
                    x, y = data_model.get_batch(b)
                    feed = {
                        self.input_data: x,
                        self.targets: y
                    }
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    _, train_loss, state, predicted_output = sess.run([self.optimizer, self.cost, self.final_state, self.predicted_output], feed)
                    accuracy = 100 * np.sum(np.equal(y, predicted_output)) / float(y.size)
                    print("{}/{} : Epoch {} - Loss = {:.3f}, Accuracy = {:.2f}".format(e * num_batches + b, args.epochs * num_batches, e, train_loss, accuracy))
                    if (e * num_batches + b) % args.save_every == 0 or (e == args.epochs - 1 and b == data_model.num_batches() - 1):
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                        print("Model saved to {}".format(checkpoint_path))

    def test(self, vocab, test_data):
        args = self.args
        init = tf.global_variables_initializer()
        test_data = list(itertools.chain.from_iterable(test_data))
        word_to_value = dict((c, i) for i, c in enumerate(vocab))
        test_value_set = np.array([word_to_value[word] for word in test_data])
        n_batches = int(len(test_data) / args.seq_length)
        limit = n_batches * args.seq_length
        data_x = np.reshape(test_value_set[:limit], [n_batches, args.seq_length])
        data_y = np.reshape(test_value_set[1:limit+1], [n_batches, args.seq_length])
        with tf.Session() as sess:
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            ppl = 0
            for i in range(n_batches):
                seq = np.reshape(data_x[i, :], [args.batch_size, args.seq_length])
                feed = {self.input_data: seq, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                prob = np.log(prob[np.arange(len(prob)), data_y[i, :]])
                ppl += np.sum(prob)
                update_progress(i+1, n_batches, 'Current Perplexity = {:.2f}'.format(np.exp(-ppl / (args.seq_length * (i + 1)))))
            ppl /= args.seq_length * n_batches
            ppl = np.exp(-ppl)
            return ppl

    def generate(self, vocab, start, num_predictions):
        args = self.args
        init = tf.global_variables_initializer()
        word_to_value = dict((word, i) for i, word in enumerate(vocab))
        value_to_word = dict((i, word) for i, word in enumerate(vocab))
        sentence = start
        with tf.Session() as sess:
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            for word in start[:-1]:
                x = np.reshape(word_to_value[word], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                state = sess.run(self.final_state, feed)

            word = start[-1]
            for i in range(num_predictions):
                x = np.reshape(word_to_value[word], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                val = int(np.searchsorted(np.cumsum(prob[0]), np.random.rand(1)))
                word = value_to_word[val]
                sentence += [word]
            return sentence
