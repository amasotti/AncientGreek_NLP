"""
Auxiliary functions

"""

from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from more_itertools import locate

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


def createCorpus(text):
    '''
    :params text - the raw text

    returns  + the corpus, a list of list with tokenized sentences
             + the vocab (a list of tokens)

    '''
    Stokenizer = TokenizeSentence('greek')
    Wtokenizer = WordTokenizer('greek')
    sentences = Stokenizer.tokenize(text)
    new_sentences = []
    vocab = []
    for sent in sentences:
        new_sent = Wtokenizer.tokenize(sent)
        new_sentences.append(new_sent)
        for w in new_sent:
            if w not in vocab:
                vocab.append(w)

    return new_sentences, vocab


