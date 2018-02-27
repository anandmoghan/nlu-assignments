from collections import Counter
import numpy as np
import json
import re


UNKNOWN = '<UNK>'
START = '<s>'
END = '</s>'

corpus_json = {
    'brown': 'brown.json',
    'gutenberg': 'gutenberg.json',
    'brown_gutenberg': 'brown_gutenberg.json'
}


def generate_sent(json_content):
    u_count = Counter(json_content['models']['unigram']['counts'])
    b_count = json_content['models']['bigram']['counts']
    b_disc = json_content['models']['bigram']['disc']
    t_count = json_content['models']['trigram']['counts']

    def absolute_disc(value):
        if value == 0:
            return 0
        elif value == 1:
            return 0.54
        else:
            return 0.75

    def get_word(tokens):
        prob = {}
        ln = len(tokens)
        most_common = dict(u_count.most_common(1000)).keys()
        for key in most_common:
            try:
                b_prob = (b_count[tokens[-1]][key] - absolute_disc(b_count[tokens[-1]][key])) / u_count[tokens[-1]]
            except KeyError:
                b_prob = (b_disc[tokens[-1]] * u_count[key]) / u_count[tokens[-1]]
            if ln > 2:
                b_prob *= 0.7
                try:
                    t_prob = t_count[tokens[-2]][tokens[-1]][key] / b_count[tokens[-1]][key]
                except KeyError:
                    t_prob = 0
            else:
                t_prob = 0
            prob[key] = 0.3 * t_prob + b_prob
        words = sorted(prob, key=prob.get, reverse=True)[:5]
        np.random.shuffle(words)
        while words[0] == START or words[0] == UNKNOWN or (words[0] == END and ln < 11) or words[0] == tokens[ln-1]:
            np.random.shuffle(words)
        return words[0]

    sentence = [START]
    i = 1
    while i < 11:
        word = get_word(sentence)
        sentence += [word]
        i += 1

    sentence = sentence[1:]
    sentence[0] = sentence[0].title()
    if sentence[-1] == END:
        sentence[-1] = '.'
    else:
        sentence += ['.']
    sentence = ' '.join(sentence)
    sentence = re.sub('\si\s', ' I ', sentence)
    sentence = re.sub('(\s\.)|(\s,\s\.)', '.', sentence)
    sentence = re.sub('\s,', ',', sentence)
    return sentence


bg_content = json.load(open(corpus_json['brown_gutenberg'], 'r'))
print(generate_sent(bg_content))
