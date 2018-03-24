import argparse
import pickle
import re
import tensorflow as tf
tf.reset_default_graph()

from models import CharacterLSTM, DataModel


def clean_text(text):
    text = text.lower()
    text = re.sub("([@]+)|([-]{2,})|(\\n)|([\s]{2,})|([\"*:\[\]()]+)", ' ', text)
    return text


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./save', help='Directory to save model checkpoints')
parser.add_argument('--start', type=str, default='the sun ', help='Start of the generation')
parser.add_argument('--predict', type=int, default=10, help='No of predictions')
args = parser.parse_args()

with open(args.save_dir + '/args.pkl', 'rb') as f:
    model_args = pickle.load(f)

model_args.batch_size = 1
model_args.seq_length = 1

with open(args.save_dir + '/character_set.pkl', 'rb') as f:
    character_set = pickle.load(f)

model = CharacterLSTM(model_args)
sentence = model.generate(character_set, clean_text(args.start), args.predict)
print(sentence)
