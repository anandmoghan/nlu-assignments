import argparse
import json
import pickle

import time
from models import CharacterLSTM, DataModel


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='./save', help='Directory to save model checkpoints')
parser.add_argument('--start', type=str, default='the', help='Start of the generation')
parser.add_argument('--predict', type=int, default=200, help='No of predictions')
args = parser.parse_args()


with open(args.save_dir + '/args.pkl', 'rb') as f:
    model_args = pickle.load(f)

with open(args.save_dir + '/character_set.pkl', 'rb') as f:
    character_set = pickle.load(f)

with open(args.save_dir + '/gutenberg.json', 'r') as f:
    json_content = json.load(f)
    test_data = json_content['test_data']

args.seq_length = 2
args.batch_size = 1
model_args.seq_length = 2
model_args.batch_size = 1

model = CharacterLSTM(model_args)
print('Calculating Perplexity...')
start_time = time.time()
ppl = model.test(character_set, test_data)
end_time = time.time()
print('Finished in %d minutes %d seconds' % ((end_time - start_time) / 60, (end_time - start_time) % 60))
print('Perplexity: %f' % ppl)
