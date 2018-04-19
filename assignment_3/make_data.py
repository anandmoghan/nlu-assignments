import nltk
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

import numpy as np

from logger import Logger

EMB_SIZE = 100
logger = Logger()


def transform_emb(model, data):
    emb_data = []
    for s in data:
        emb_sent = []
        for word in s:
            try:
                emb_sent.append(model.wv[word])
            except KeyError:
                emb_sent.append(np.zeros(EMB_SIZE))
        emb_data.append(emb_sent)
    return emb_data


def make_true_test(test_tags):
    with open('./splits/true_test.txt', 'w') as tf:
        for i, s in enumerate(test_tags):
            for j, w in enumerate(s):
                tf.write('{}\n'.format(w))
            tf.write('\n')


def make_simple(train_sentences, test_sentences, train_tags):
    logger.start_timer('Making Simple Splits..')
    with open('./splits/simple_train.txt', 'wb') as tf:
        for i, s in enumerate(train_sentences):
            for j, w in enumerate(s):
                tf.write('{} {}\n'.format(w, train_tags[i][j]).encode())
            tf.write('\n'.encode())

    with open('./splits/simple_test.txt', 'wb') as tf:
        for i, s in enumerate(test_sentences):
            for j, w in enumerate(s):
                tf.write('{}\n'.format(w).encode())
            tf.write('\n'.encode())
    logger.end_timer()


def make_emb(train_sentences, test_sentences, train_tags):
    logger.start_timer('Making Embedding Splits..')
    word2vec = Word2Vec(sentences=train_sentences, size=EMB_SIZE, window=5, min_count=1, workers=6)
    embedded_train = transform_emb(word2vec, train_sentences)
    embedded_test = transform_emb(word2vec, test_sentences)
    with open('./splits/emb_train.txt', 'wb') as tf:
        for i, s in enumerate(embedded_train):
            for j, w in enumerate(s):
                tf.write('{} {}\n'.format(' '.join(str(d) for d in w), train_tags[i][j]).encode())
            tf.write('\n'.encode())

    with open('./splits/emb_test.txt', 'wb') as tf:
        for i, s in enumerate(embedded_test):
            for j, w in enumerate(s):
                tf.write('{}\n'.format(' '.join(str(d) for d in w)).encode())
            tf.write('\n'.encode())
    logger.end_timer()


def make_simple_emb(train_sentences, test_sentences, train_tags):
    logger.start_timer('Making Simple and Embedding Splits..')
    word2vec = Word2Vec(sentences=train_sentences, size=EMB_SIZE, window=5, min_count=1, workers=6)
    embedded_train = transform_emb(word2vec, train_sentences)
    embedded_test = transform_emb(word2vec, test_sentences)
    with open('./splits/simple_emb_train.txt', 'wb') as tf:
        for i, s in enumerate(embedded_train):
            for j, w in enumerate(s):
                tf.write('{} {} {}\n'.format(train_sentences[i][j], ' '.join(str(d) for d in w), train_tags[i][j]).encode())
            tf.write('\n'.encode())

    with open('./splits/simple_emb_test.txt', 'wb') as tf:
        for i, s in enumerate(embedded_test):
            for j, w in enumerate(s):
                tf.write('{} {}\n'.format(test_sentences[i][j], ' '.join(str(d) for d in w)).encode())
            tf.write('\n'.encode())
    logger.end_timer()


def make_simple_pos(train_sentences, test_sentences, train_tags):
    logger.start_timer('Making Simple POS Splits..')
    with open('./splits/simple_pos_train.txt', 'wb') as tf:
        for i, s in enumerate(train_sentences):
            pos_tags = nltk.pos_tag(s)
            for j, w in enumerate(s):
                tf.write('{} {} {}\n'.format(w, pos_tags[j], train_tags[i][j]).encode())
            tf.write('\n'.encode())

    with open('./splits/simple_pos_test.txt', 'wb') as tf:
        for i, s in enumerate(test_sentences):
            pos_tags = nltk.pos_tag(s)
            for j, w in enumerate(s):
                tf.write('{} {}\n'.format(w, pos_tags[j]).encode())
            tf.write('\n'.encode())
    logger.end_timer()


sentences = []
labels = []
with open('./ner.txt', 'rb') as f:
    sent = []
    sent_labels = []
    for line in f:
        line = np.unicode(line, 'utf-8').strip()
        if line == '':
            sentences += [sent]
            labels += [sent_labels]
            sent = []
            sent_labels = []
        else:
            line = line.split(' ')
            sent += [line[0]]
            sent_labels += [line[1]]

sentences = np.array(sentences)
labels = np.array(labels)

splits = train_test_split(sentences, labels, test_size=0.2)
make_true_test(splits[3])
make_simple(splits[0], splits[1], splits[2])
make_emb(splits[0], splits[1], splits[2])
make_simple_emb(splits[0], splits[1], splits[2])
make_simple_pos(splits[0], splits[1], splits[2])
