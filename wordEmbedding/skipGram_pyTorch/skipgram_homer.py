from wordEmbedding.utils.dataset import skip_gram_dataset
from argparse import Namespace
import json
import numpy as np

args = Namespace(
    corpus = "../../data/Homer_tokenized_corpus.npy",
    w2i = "../../data/vocabularies/Homer_word2index.json",
    skipDataset = "../../data/vocabularies/Homer_skipgram_dataset"

)
# Load tokenized corpus (see preprocessing.py in utils)
corpus = np.load(args.corpus, allow_pickle=True)
corpus = corpus.tolist()

# Load the Word2Index dictionary (see preprocessing.py in utils))
with open(args.w2i, "r", encoding="utf-8") as fp:
    w2i = json.load(fp)

# Create a reverse lookup table
index2word = {i: w for w, i in w2i.items()}

# extract the dataset for training
skipDataset = skip_gram_dataset(corpus=corpus, word2index=w2i, window=7)
# save it for later use
np.save(args.skipDataset,skipDataset,allow_pickle=True)



