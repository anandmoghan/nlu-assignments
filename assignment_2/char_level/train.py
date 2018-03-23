import argparse
import pickle

import time
from char_models import CharacterLSTM, DataModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./save', help='Directory to save model checkpoints')
parser.add_argument('--rnn_layers', type=int, default=2, help='No:of layers in the RNN')
parser.add_argument('--rnn_size', type=int, default=128, help='Size of RNN hidden states')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate')
parser.add_argument('--decay', type=float, default=0.97, help='Decay Rate')
parser.add_argument('--batch_size', type=int, default=100, help='Mini-batch size')
parser.add_argument('--seq_length', type=int, default=50, help='Sequence Length')
parser.add_argument('--epochs', type=int, default=50, help='No:of Epochs')
parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
args = parser.parse_args()

with open('character_set.pkl', 'rb') as f:
    character_set = pickle.load(f)

data_model = DataModel(args.batch_size, args.seq_length, character_set, args.save_dir)
args.vocab_size = data_model.get_total_characters()

with open(args.save_dir + '/args.pkl', 'wb') as f:
    pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)


model = CharacterLSTM(args)
start_time = time.time()
model.train(data_model)
end_time = time.time()
print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))
