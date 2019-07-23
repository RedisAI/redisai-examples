import json
from os.path import dirname
import re

import numpy as np

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:
    def __init__(self):
        pass

    def load_data(self, path):
        with open(path) as f:
            self.__dict__ = json.load(f)


voc = Voc()
voc_file = f'{dirname(dirname(dirname(__file__)))}/models/pytorch/chatbot/voc.json'
voc.load_data(voc_file)


def normalize_string(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def get_batched_indices(sentence):
    return [[el for el in [voc.word2index.get(word) for word in sentence.split(' ')] if el is not None] + [EOS_token]]
    # return [[voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]]


def list2numpy(indices):
    return np.array(indices, dtype=np.long).transpose()


def get_length(array):
    return np.array([len(array)], dtype=np.long)


def indices2str(indices):
    return ' '.join([voc.index2word[str(ind)] for ind in indices])
