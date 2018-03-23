import argparse
import pickle
import re

from models import TokenLSTM, DataModel


def clean_text(text):
    text = re.sub('([\d]+)|(\'\')|(\s[\']+)|[";*:`~$]+|(--)|([\s]+-[\s]+)|[()\[\]]', ' ', text)
    text = text.lower()
    return text


def tokenize(text):
    return re.split('[\s]+', text)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./save', help='Directory to save model checkpoints')
parser.add_argument('--start', type=str, default='the', help='Start of the generation')
parser.add_argument('--predict', type=int, default=20, help='No of predictions')
args = parser.parse_args()

with open(args.save_dir + '/args.pkl', 'rb') as f:
    model_args = pickle.load(f)

with open(args.save_dir + '/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


model = TokenLSTM(model_args, training=False)
start = clean_text(args.start)
sentence = model.generate(vocab, tokenize(start), args.predict)
print(' '.join(sentence))
