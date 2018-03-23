from tensorflow.contrib import rnn, legacy_seq2seq
import tensorflow as tf
import numpy as np
import pickle
import nltk
import time
import json
import re
import os

from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits


class DataModel:
    UNKNOWN = '@'

    def __init__(self, batch_size, seq_length, character_set, save_dir, train_percent=0.8, dev_percent=0.1):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.save_dir = save_dir
        exp = '[^{}]'.format(''.join(character_set))

        print('Pre-processing...')
        start_time = time.time()
        files = nltk.corpus.gutenberg.fileids()
        files = ['austen-sense.txt']
        train_data = dev_data = test_data = ''
        for file in files:
            data = nltk.corpus.gutenberg.raw(file)
            f_train_data, f_dev_data, f_test_data = self.split_data(self.clean_text(data, exp), train_percent, dev_percent)
            train_data += ' '.join(f_train_data)
            dev_data += ' '.join(f_dev_data)
            test_data += ' '.join(f_test_data)

        json_content = json.dumps({
            'train_data': train_data,
            'dev_data': dev_data,
            'test_data': test_data
        }, indent=4)
        with open(self.get_file_name('gutenberg.json'), 'w') as f:
            f.write(json_content)

        self.character_set = sorted(set(train_data))
        self.total_characters = len(self.character_set)
        self.train_len = len(train_data)
        self.char_to_value = dict((c, i) for i, c in enumerate(self.character_set))
        self.character_value_set = np.array([self.char_to_value[c] for c in train_data])
        self.save_object(self.character_set, self.get_file_name('character_set.pkl'))
        self.save_object(self.char_to_value, self.get_file_name('char_to_value.pkl'))
        np.save(self.get_file_name('character_value_set.npy'), self.character_value_set)
        print('Characters: %s' % self.character_set)
        print('Total Unique Characters: %d' % self.total_characters)
        print('Total Characters: %d' % self.train_len)

        self.n_batches = int((len(self.character_value_set) - 1) / (batch_size * seq_length))
        limit = self.n_batches * batch_size * seq_length
        data_x = self.character_value_set[:limit]
        data_y = self.character_value_set[1:limit + 1]
        self.batch_x = np.split(np.reshape(data_x, [self.batch_size, -1]), self.n_batches, 1)
        self.batch_y = np.split(np.reshape(data_y, [self.batch_size, -1]), self.n_batches, 1)
        end_time = time.time()
        print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))

    def clean_text(self, text, exp):
        text = text.lower()
        text = re.sub("([-]{2,})|(\\n)", ' ', text)
        text = re.sub(exp, self.UNKNOWN, text)
        text = nltk.tokenize.sent_tokenize(text)
        return text

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

    def get_total_characters(self):
        return self.total_characters

    def num_batches(self):
        return self.n_batches

    def get_batch(self, b):
        return self.batch_x[b], self.batch_y[b]


class CharacterLSTM:
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
        # loss = legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self.targets, [-1])], [tf.ones([args.batch_size * args.seq_length])])
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

    def test(self, character_set, test_data):
        args = self.args
        init = tf.global_variables_initializer()
        char_to_value = dict((c, i) for i, c in enumerate(character_set))
        test_value_set = np.array([char_to_value[c] for c in test_data])
        with tf.Session() as sess:
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            ppl = 0
            n_groups = int(len(test_value_set) / args.seq_length)
            test_seq = test_value_set[:n_groups * args.seq_length]
            test_seq = np.reshape(test_seq, [n_groups, args.seq_length])
            for i in range(n_groups - 1):
                seq = np.reshape(test_seq[i, :], [args.batch_size, args.seq_length])
                feed = {self.input_data: seq, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                prob = np.log(prob[np.arange(len(prob)), test_seq[i+1, :]])
                ppl += np.sum(prob)
            ppl /= (args.seq_length * (n_groups - 1))
            ppl = np.exp(-ppl)
            return ppl

    def generate(self, character_set, start, num_predictions):
        args = self.args
        init = tf.global_variables_initializer()
        char_to_value = dict((c, i) for i, c in enumerate(character_set))
        value_to_char = dict((i, c) for i, c in enumerate(character_set))
        sentence = start
        with tf.Session() as sess:
            sess.run(init)
            tf_saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.get_checkpoint_state(args.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            state = sess.run(self.cell.zero_state(1, tf.float32))
            for c in start[:-1]:
                x = np.reshape(char_to_value[c], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                state = sess.run(self.final_state, feed)

            c = start[-1]
            for i in range(num_predictions):
                x = np.reshape(char_to_value[c], [1, 1])
                feed = {self.input_data: x, self.initial_state: state}
                prob, state = sess.run([self.probs, self.final_state], feed)
                if c == ' ':
                    val = int(np.searchsorted(np.cumsum(prob[0]), np.random.rand(1)))
                else:
                    val = int(np.argmax(prob[0]))
                c = value_to_char[val]
                sentence += c
            return sentence
