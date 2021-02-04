"""
Short PyTorch implementation of the Word2Vec

"""

__author__ = "Antonio Masotti"

from argparse import Namespace
from modules import *
from utils import *


# Paths (I hate to have constantly to write paths...)
args = Namespace(
    raw_data='../../data/raw_data/HomerGesamt_deaccented.txt',
    model='../../data/models/torch-w2vec.model'
)


# Load data
with open(args.raw_data, 'r', encoding='utf-8') as src:
    data = src.read()


corpus, vocab = createCorpus(data)
vocab_size = len(vocab)

word2index = {w: i for i, w in enumerate(vocab)}
index2word = {i: w for w, i in word2index.items()}

# Parameters & settings
param = Namespace(
    train=int(len(vocab) * 0.7),
    val=int(len(vocab) * 0.15),
    test=int(len(vocab) * 0.15),
    window=9,
    embeddings = 100,
    lr = 0.02,
    epochs = 20
)

cbow_dataset = createCbow(corpus, vocab,param.window)

print(cbow_dataset[0])
print(cbow_dataset[0])

