"""
Auxiliary functions
# Inspired by https://srijithr.gitlab.io/post/word2vec/
"""
import json

from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from more_itertools import locate
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

def allIndices(sent, wanted):
    '''
    Find all indices of a given word in a sentence (needed, if a word occurs more than once in a sent)
    '''
    indices = list(locate(sent, lambda a : a == wanted))
    return indices


def find_neighbours(corpus, target, window=9):
    '''
    find the context of a word in a given corpus
    '''
    print('Finding the neighbours for each word in vocab')
    context = []
    # iterate per sentence
    for sentence in corpus:
        # if the wanted form is in the sentence, extract all the indices
        if target in sentence:
            indices = allIndices(sentence, target)

            # for each occurrence
            for occurrence in indices:
                for j in range(max(occurrence - window, 0), min(occurrence + window, len(sentence))):
                    if j != occurrence:
                        context.append(sentence[j])
    return context


def createCbow(corpus, vocab, window):
    '''

    Builds the dataset for cbow

    '''
    print('Building the CBOW dataset...')
    data = []
    for word in vocab:
        context = find_neighbours(corpus, word, window)
        data.append((context, word))
    return data


def createCorpus(text, save=True):
    '''
    :params text - the raw text

    returns  + the corpus, a list of list with tokenized sentences
             + the vocab (a dictionary with the frequency of the tokens scaled by the total number of words.

    '''
    Stokenizer = TokenizeSentence('greek')
    Wtokenizer = WordTokenizer('greek')
    sentences = Stokenizer.tokenize(text)
    new_sentences = []
    vocab = dict()
    print('Building corpus and freqDictionary')
    for sent in tqdm(sentences, desc="Sentences"):
        new_sent = Wtokenizer.tokenize(sent)
        new_sentences.append(new_sent)
        for w in new_sent:
            if w not in vocab:
                vocab[w] = 1
            else:
                vocab[w] += 1

    vocab_size = len(vocab)
    for k, v in vocab.items():
        # Subsampling, see paper by Goldberg & Levy
        frac = v / vocab_size
        p_w = (1+np.sqrt(frac * 0.001)) * 0.001 / frac
        # update the value for the word
        vocab[k] = p_w
    if save:
        print('Saving the frequencies')
        with open('../../data/vocabularies/Homer_word_frequencies.json', 'w', encoding='utf-8') as fp:
            json.dump(vocab, fp, ensure_ascii=False)
        print('Saving the corpus')
        arr = np.array(new_sentences, dtype=object)
        np.save('../../data/Homer_tokenized_corpus.npy', arr)
    return new_sentences, vocab


