from keras.layers import LSTM, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from keras.utils import to_categorical
from keras.models import Model, Input
from keras_contrib.layers import CRF
from itertools import chain

# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from argparse import ArgumentParser


MAX_LEN = 60


parser = ArgumentParser()
parser.add_argument('--test', action='store_true', help='Test Data')
args = parser.parse_args()


def save_object(file_name, obj):
    with open(file_name, 'wb') as pf:
        pickle.dump(obj, pf, pickle.HIGHEST_PROTOCOL)


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


tokens = chain.from_iterable(sentences)
unique_tokens = list(set(tokens))
vocab_len = len(unique_tokens)
token_to_index = dict([(w, i + 1) for i, w in enumerate(unique_tokens)])

tags = ['O', 'D', 'T']
tags_len = len(tags)
tags_to_index = dict([(t, i) for i, t in enumerate(tags)])

encoded_sentences = [[token_to_index[t] for t in sent] for sent in sentences]
encoded_tags = [[tags_to_index[t] for t in sent] for sent in labels]

encoded_sentences = pad_sequences(sequences=encoded_sentences, maxlen=MAX_LEN, padding='post', value=0)
encoded_tags = pad_sequences(sequences=encoded_tags, maxlen=MAX_LEN, padding='post', value=tags_to_index['O'])

train_sentences, test_sentences, train_tags, test_tags = train_test_split(encoded_sentences, encoded_tags, test_size=0.1)
train_tags = [to_categorical(i, num_classes=tags_len) for i in train_tags]

input_layer = Input(shape=(MAX_LEN,))
embedding_layer = Embedding(input_dim=vocab_len + 1, output_dim=200, input_length=MAX_LEN, mask_zero=True)(input_layer)
lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.2))(embedding_layer)
crf = CRF(tags_len)
crf_output = crf(lstm_layer)

model = Model(input_layer, crf_output)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
print(model.summary())

if not args.test:
    history = model.fit(train_sentences, np.array(train_tags), batch_size=64, epochs=5, validation_split=0.1, verbose=1)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("new_model.h5")
    print("Saved model to disk")

    '''
    hist = pd.DataFrame(history.history)
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 12))
    plt.plot(hist["acc"], 'b', label='Train Accuracy')
    plt.plot(hist["val_acc"], 'r', label='Validation Accuracy')
    plt.legend()
    plt.savefig('plot.png')'''

model.load_weights('new_model.h5')
test_pred = model.predict(test_sentences)
test_pred = np.argmax(test_pred, axis=2)

test_pred = np.reshape(test_pred, [-1, ])
test_tags = np.reshape(test_tags, [-1, ])

print('Accuracy Score', accuracy_score(test_tags, test_pred))
print('Macro F1 Score', f1_score(test_tags, test_pred, average='macro'))
print('Micro F1 Score', f1_score(test_tags, test_pred, average='micro'))
print('Weighted F1 Score', f1_score(test_tags, test_pred, average='weighted'))
print('Precision Score', precision_score(test_tags, test_pred, average='weighted'))
print('Recall Score', recall_score(test_tags, test_pred, average='weighted'))
print('Confusion Matrix', confusion_matrix(test_tags, test_pred))
