# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'


# imports
import os
from argparse import Namespace
import nltk.data
import pandas as pd
import re
from auxiliary.GreekToVec_utils import delete_stopwords,load_file
from cltk.tokenize.sentence import TokenizeSentence

param = Namespace(
    raw_text = '../data/HomerGesamt_cleaned.txt',
    stopwords = '../data/stopwords.txt',
    window = 15, # quite high but useful for semantic analysis
    train_prop = 0.7,
    val_prop = 0.15,
    test_prop = 0.15,
    output = '../data/Homer_cbow_preprocessed.csv',
    MASK = "<SENT_BOUND>"
)

# load file
homer = load_file(param.raw_text)

# Sentence tokenizer
greek_tokenizer = TokenizeSentence('greek')
homer_sentences = greek_tokenizer.tokenize(homer)


# clean tokens
def clean_delete_stopwords(sentences):
    '''

    :param sentences: a list of sentences
    :return: the same list whitout stopwords and with spacing after punctuation
    '''
    new_sentences = []
    for s in sentences:
        s = re.sub(r"([.,!?])", r" \1 ", s)
        tokens = delete_stopwords(stopwords_file=param.stopwords, text=s)
        tokens = ' '.join(w for w in tokens)
        new_sentences.append(tokens)
    del(sentences)
    return new_sentences

homer_sentences = clean_delete_stopwords(homer_sentences)


# window creator

flatten = lambda outer : [item for inner in outer for item in inner]
windows = flatten([list(nltk.ngrams([param.MASK] * param.window + s.split(' ') + [param.MASK] * param.window, param.window*2 + 1 )) for s in homer_sentences])

# initialize dataframe
data = []
for window in windows:
    target = window[param.window]
    context = []
    for i, token in enumerate(window):
        if token == param.MASK or i == param.window:
            continue
        else:
            context.append(token)
    if len(context) > 2:
        data.append([" ".join(t for t in context), target])

# convert in pandas dataframe
cbow_homer = pd.DataFrame(data, columns = ["context","word"])

# create split
cbow_len = len(cbow_homer)

def split_cbow(row_num):
    if row_num <= cbow_len * param.train_prop:
        return 'train'
    elif (row_num > cbow_len*param.train_prop) and (row_num <= cbow_len * param.train_prop + cbow_len * param.val_prop):
        return 'validation'
    else:
        return 'test'

cbow_homer['split_subset'] = cbow_homer.apply(lambda row : split_cbow(row.name), axis = 1)

print(cbow_homer.head(10))

# save
cbow_homer.to_csv(param.output, index=False)

