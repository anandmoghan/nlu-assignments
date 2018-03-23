from tensorflow.contrib import rnn, legacy_seq2seq
from collections import Counter
import tensorflow as tf
import numpy as np
import itertools
import pickle
import nltk
import time
import json
import re
import os


class DataModel:
    UNKNOWN = '<unk>'
    MIN_WORD_COUNT = 1

    def __init__(self, batch_size, seq_length, save_dir, train_percent=0.8, dev_percent=0.1):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.save_dir = save_dir

        print('Pre-processing...')
        start_time = time.time()
        train_data = dev_data = test_data = []
        files = nltk.corpus.gutenberg.fileids()
        files = ['austen-sense.txt']
        for file in files:
            data = nltk.corpus.gutenberg.sents(file)
            f_train_data, f_dev_data, f_test_data = self.split_data(data, train_percent, dev_percent)
            train_data += self.clean_tokenize(f_train_data)
            dev_data += self.clean_tokenize(f_dev_data)
            test_data += self.clean_tokenize(f_test_data)

        words = itertools.chain.from_iterable(train_data)

        w_counter = Counter(words)
        train_data = self.put_unknown(train_data, w_counter)
        dev_data = self.put_unknown(train_data, w_counter)
        test_data = self.put_unknown(train_data, w_counter)

        print(train_data)

        json_content = json.dumps({
            'train_data': train_data,
            'dev_data': dev_data,
            'test_data': test_data
        }, indent=4)
        with open(self.get_file_name('gutenberg.json'), 'w') as f:
            f.write(json_content)

        print(train_data)

        self.vocab = sorted(set(train_data))
        self.vocab_size = len(self.vocab)
        self.train_len = len(train_data)
        self.word_to_value = dict((word, i) for i, word in enumerate(self.vocab))
        self.word_value_set = np.array([self.word_to_value[word] for word in train_data])
        self.save_object(self.vocab, self.get_file_name('vocab.pkl'))
        self.save_object(self.word_to_value, self.get_file_name('word_to_value.pkl'))
        np.save(self.get_file_name('word_value_set.npy'), self.word_value_set)
        print('Total Unique Words: %d' % self.vocab_size)
        print('Total Words: %d' % self.train_len)

        self.n_batches = int((len(self.word_value_set) - 1) / (batch_size * seq_length))
        limit = self.n_batches * batch_size * seq_length
        data_x = self.word_value_set[:limit]
        data_y = self.word_value_set[1:limit + 1]
        self.batch_x = np.split(np.reshape(data_x, [self.batch_size, -1]), self.n_batches, 1)
        self.batch_y = np.split(np.reshape(data_y, [self.batch_size, -1]), self.n_batches, 1)
        end_time = time.time()
        print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))

    def put_unknown(self, data, w_counter):
        return [[self.UNKNOWN if w_counter[word] < self.MIN_WORD_COUNT else word for word in sent] for sent in data]

    @staticmethod
    def clean_text(text):
        text = re.sub('([\d]+)|(\'\')|(\s[\']+)|[";*:`~$]+|(--)|([\s]+-[\s]+)|[()\[\]]|([\n])', ' ', text)
        text = text.lower()
        return nltk.tokenize.sent_tokenize(text)

    @staticmethod
    def clean_tokenize(text):
        sentences = []
        for sent in text:
            words = ['<s>'] + sent
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

        cells = []
        for _ in range(args.rnn_layers):
            cells.append(rnn.BasicLSTMCell(args.rnn_size))

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, self.final_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, args.seq_length])

        self.lr = tf.Variable(0.0, trainable=False)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.targets, [-1])], [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / (args.batch_size * args.seq_length)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def get_file_name(self, file_name):
        return self.save_dir + '/' + file_name

    def train(self, data_model):
        args = self.args
        init = tf.global_variables_initializer()
        num_batches = data_model.num_batches()
        tf_saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(init)
            for e in range(args.epochs):
                sess.run(tf.assign(self.lr, args.lr * (args.decay ** e)))
                state = sess.run(self.initial_state)
                for b in range(num_batches):
                    x, y = data_model.get_batch(b)
                    feed = {
                        self.input_data: x,
                        self.targets: y
                    }
                    for i, (c, h) in enumerate(self.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    _, train_loss, state, predicted_output = sess.run([self.optimizer, self.cost, self.final_state, self.predicted_output], feed)
                    accuracy = np.sum(np.equal(y, predicted_output)) / float(y.size)
                    print("{}/{} - Epoch {}, Loss = {:.3f}, Accuracy = {}".format(e * num_batches + b, args.epochs * num_batches, e, train_loss, accuracy))
                    if (e * num_batches + b) % args.save_every == 0 or (e == args.epochs - 1 and b == data_model.num_batches() - 1):
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                        print("Model saved to {}".format(checkpoint_path))

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
