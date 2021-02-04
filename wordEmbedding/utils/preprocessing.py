"""
Build vocab & corpus and save them for later use

"""

__author__ = "Antonio Masotti"

import json
from argparse import Namespace
from utils import *


# Paths (I hate to have constantly to write paths...)
args = Namespace(
    raw_data='../../data/raw_data/HomerGesamt_deaccented.txt',
    model='../../data/models/torch-w2vec.model'
)


# Load data
with open(args.raw_data, 'r', encoding='utf-8') as src:
    data = src.read()

# tokenize and build corpus and freqDict from text
corpus, freqVocab = createCorpus(data)

# vocabulary size
vocab_size = len(freqVocab)

# Lookup tables
word2index = {w: i for i, w in enumerate(freqVocab.keys())}
index2word = {i: w for w, i in word2index.items()}

with open('../../data/vocabularies/Homer_word2index.json', 'w', encoding='utf-8') as fp:
    json.dump(word2index, fp, ensure_ascii=False)


print(vocab_size)
