import argparse
import pickle

from models import TokenLSTM, DataModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./save', help='Directory to save model checkpoints')
parser.add_argument('--rnn_layers', type=int, default=2, help='No:of layers in the RNN')
parser.add_argument('--rnn_size', type=int, default=128, help='Size of RNN hidden states')
parser.add_argument('--lr', type=float, default=0.002, help='Learning Rate')
parser.add_argument('--decay', type=float, default=0.97, help='Decay Rate')
parser.add_argument('--batch_size', type=int, default=100, help='Mini-batch size')
parser.add_argument('--seq_length', type=int, default=10, help='Sequence Length')
parser.add_argument('--epochs', type=int, default=50, help='No:of Epochs')
parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
args = parser.parse_args()

data_model = DataModel(args.batch_size, args.seq_length, args.save_dir)
args.vocab_size = data_model.get_vocab_size()

with open(args.save_dir + '/args.pkl', 'wb') as f:
    pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

model = TokenLSTM(args)
model.train(data_model)
