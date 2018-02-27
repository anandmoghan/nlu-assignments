from collections import defaultdict
from collections import Counter
import numpy as np
import nltk
import json
import time
import re

train_percent = 0.8
dev_percent = 0.1
UNKNOWN = '<UNK>'

corpus_data = {
    'brown': nltk.corpus.brown,
    'gutenberg': nltk.corpus.gutenberg
}
corpus_json = {
    'brown': 'brown.json',
    'gutenberg': 'gutenberg.json',
    'brown_gutenberg': 'brown_gutenberg.json'
}
data_type = {
    'brown': ['brown'],
    'gutenberg': ['gutenberg'],
    'brown_gutenberg': ['brown', 'gutenberg']
}
nltk.download('brown')
nltk.download('gutenberg')


def clean_text(text):
    text = re.sub('([\d]+)|(\'\')|(\s[\']+)|[";*:`~$]+|(--)|([\s]+-[\s]+)|[()\[\]]', ' ', text)
    text = re.sub('[\s]+[.?!]+', ' </s>_<s> ', text)
    text += '<s> '
    text = text.lower()
    text = re.split('_', text)
    text = text[0:-2]
    return text


def split_data(sentences):
    np.random.shuffle(sentences)
    total_length = len(sentences)
    train_set = int(train_percent * total_length)
    dev_set = int(train_set + dev_percent * total_length)
    return sentences[0:train_set], sentences[train_set:dev_set], sentences[dev_set:-1]


def tokenize(text):
    return re.split('[\s]+', text)


def absolute_disc(value):
    if value == 0:
        return 0
    elif value == 1:
        return 0.54
    else:
        return 0.75


def n_gram(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


def get_key(t):
    return '_'.join(t)


def train(sentences, corpus):
    train_data, dev_data, test_data = split_data(sentences)
    train_data = ' '.join(train_data)
    tokens = tokenize(train_data)
    print('Calculating counts.')
    u_counter = Counter(tokens)
    b_counter = Counter(n_gram(tokens, 2))
    t_counter = Counter(n_gram(tokens, 3))

    print('Calculating discounts.')
    u_counter[UNKNOWN] = sum([absolute_disc(value) for key, value in u_counter.most_common()])
    total_u_count = sum(u_counter.values())
    vocab_count = len(u_counter.keys())
    b_count = defaultdict(dict)
    b_disc = dict()
    for keys in b_counter.keys():
        b_count[keys[0]][keys[1]] = b_counter[keys]
        try:
            b_disc[keys[0]] += absolute_disc(b_counter[keys])
        except KeyError:
            b_disc[keys[0]] = absolute_disc(b_counter[keys])

    b_disc = dict((key, b_disc[key] / (total_u_count - sum([u_counter[x] for x in b_count[key].keys()]))) for key in b_disc)
    b_disc[UNKNOWN] = u_counter[UNKNOWN] / total_u_count

    t_count = defaultdict(dict)
    for keys in t_counter.keys():
        try:
            t_count[keys[0]][keys[1]][keys[2]] = t_counter[keys]
        except KeyError:
            t_count[keys[0]][keys[1]] = dict()
            t_count[keys[0]][keys[1]][keys[2]] = t_counter[keys]

    print('Saving to %s..' % corpus_json[corpus])
    n_gram_data = defaultdict(dict)
    n_gram_data['unigram']['t_count'] = total_u_count
    n_gram_data['unigram']['counts'] = dict(u_counter.most_common())
    n_gram_data['bigram']['t_count'] = sum(b_counter.values())
    n_gram_data['bigram']['counts'] = b_count
    n_gram_data['bigram']['disc'] = b_disc
    n_gram_data['trigram']['counts'] = t_count
    content = defaultdict(dict)
    content['data']['dev'] = ' '.join(dev_data)
    content['data']['test'] = ' '.join(test_data)
    content['vocab_count'] = vocab_count
    content['models'] = n_gram_data

    json_content = json.dumps(content, indent=4)
    try:
        with open(corpus_json[corpus], 'w') as f:
            f.write(json_content)
    except IOError:
        pass
    return content


def make_train(train_type):
    start_time = time.time()
    print('\nTraining with %s corpus.' % train_type)
    print('Pre-processing data.')
    sentences = []
    for key in data_type[train_type]:
        data = ' '.join(corpus_data[key].words())
        sentences += clean_text(data)
    content = train(sentences, train_type)
    end_time = time.time()
    print('Finished in %s minutes %s seconds.' % (int((end_time - start_time) / 60), int((end_time - start_time) % 60)))
    return content


def train_check(corpus):
    retrain = 'Y'
    try:
        with open(corpus_json[corpus], 'r') as f:
            json_file = f.read()
    except (ValueError, FileNotFoundError):
        json_file = {}

    if json_file != {}:
        retrain = input('Retrain %s Corpus? (Y/N): ' % corpus)

    if retrain == 'Y' or retrain == 'y':
        return True
    else:
        return False


def get_prob(json_content, tokens):
    u_count = json_content['models']['unigram']['counts']
    b_count = json_content['models']['bigram']['counts']
    b_disc = json_content['models']['bigram']['disc']
    total_u_count = json_content['models']['unigram']['t_count']
    u_prob = []
    b_prob = [0.00000001]
    for i in range(0, len(tokens)):
        word = tokens[i]
        try:
            u_prob += [(u_count[word] - absolute_disc(u_count[word])) / total_u_count]
        except KeyError:
            tokens[i] = word = UNKNOWN
            u_prob += [u_count[UNKNOWN] / total_u_count]
        if i > 0:
            try:
                b_prob += [(b_count[tokens[i - 1]][word] - (
                    absolute_disc(b_count[tokens[i - 1]][word]) if word != UNKNOWN else 0)) / u_count[tokens[i - 1]]]
            except KeyError:
                b_prob += [(b_disc[tokens[i - 1]] * u_count[word]) / u_count[tokens[i - 1]]]
    return u_prob, b_prob


def dev(json_content, corpus):
    start_time = time.time()
    print('\nOptimizing model for %s corpus on dev set.' % corpus)
    dev_data = json_content['data']['dev']
    tokens = tokenize(dev_data)
    u_prob, b_prob = get_prob(json_content, tokens)

    min_lam = 0
    min_pp = 0
    for lam in np.linspace(0, 1, 30):
        pp = sum([np.math.log(lam * b + (1 - lam) * u) for u, b in zip(u_prob, b_prob)])
        pp /= len(tokens)
        pp = np.exp(-pp)
        if min_pp == 0 or pp < min_pp:
            min_pp = pp
            min_lam = lam

    print("Min Perplexity = %.2f" % min_pp)
    print("Saving the optimal model.")
    json_content['models']['lambda'] = min_lam
    json_content = json.dumps(json_content)
    with open(corpus_json[corpus], 'w') as f:
        f.write(json_content)
    end_time = time.time()
    print('Finished in %s minutes %s seconds.' % (int((end_time - start_time) / 60), int((end_time - start_time) % 60)))


def test(train_json_content, test_json_content, train_corpus, test_corpus):
    print('\nUsing model on %s corpus to test on %s corpus.' % (train_corpus, test_corpus))
    test_data = test_json_content['data']['test']
    lam = train_json_content['models']['lambda']
    tokens = tokenize(test_data)
    u_prob, b_prob = get_prob(train_json_content, tokens)
    pp = sum([np.math.log(lam * b + (1 - lam) * u) for u, b in zip(u_prob, b_prob)])
    pp /= len(tokens)
    pp = np.exp(-pp)
    print('Perplexity: %0.2f' % pp)


b_check = train_check('brown')
g_check = train_check('gutenberg')

if b_check:
    b_content = make_train('brown')
    dev(b_content, 'brown')
else:
    b_content = json.load(open(corpus_json['brown'], 'r'))

if g_check:
    g_content = make_train('gutenberg')
    dev(g_content, 'gutenberg')
else:
    g_content = json.load(open(corpus_json['gutenberg'], 'r'))

if b_check or g_check:
    bg_content = make_train('brown_gutenberg')
    dev(bg_content, 'brown_gutenberg')
else:
    bg_content = json.load(open(corpus_json['brown_gutenberg'], 'r'))


test(b_content, b_content, 'brown', 'brown')
test(g_content, g_content, 'gutenberg', 'gutenberg')
test(bg_content, b_content, 'brown_gutenberg', 'brown')
test(bg_content, g_content, 'brown_gutenberg', 'gutenberg')
