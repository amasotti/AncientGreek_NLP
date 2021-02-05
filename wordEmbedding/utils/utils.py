"""
Auxiliary functions
# Inspired by https://srijithr.gitlab.io/post/word2vec/
"""
import json

import torch
from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
import numpy as np
from tqdm import tqdm


def createCorpus(text, save=True):
    '''
    :params text - the raw text

    returns  + the corpus, a list of list with tokenized sentences
             + the vocab (a dictionary with the frequency of the tokens scaled by the total number of words.

    '''
    with open('../../data/stopwords.txt','r',encoding="UTF-8") as src:
        stopwords = src.read()

    stopwords = stopwords.split('\n')
    stopwords.extend([".",",","?","!","-",":",";","·"])

    Stokenizer = TokenizeSentence('greek')
    Wtokenizer = WordTokenizer('greek')
    sentences = Stokenizer.tokenize(text)
    new_sentences = []
    vocab = dict()
    print('Building corpus and freqDictionary')
    for sent in tqdm(sentences, desc="Sentences"):
        new_sent = Wtokenizer.tokenize(sent)
        # Stopword deletion
        new_sent = [w for w in new_sent if w not in stopwords]
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


def save_model(model, epoch, losses,fp):
    """
    Compare the actual and the last loss value. If the value improved, save the model
    """
    if epoch > 1:
        print("Check if the model should be saved:")
        if losses[-1] < losses[-2]:
            print("Loss improved, save the model")
            torch.save({'model_state_dict': model.state_dict(),
                        'losses': losses}, fp)


# TEST
def nearest_word(inp, emb, top=5, debug=False):
    #TODO: Use cosine distance instead of euclidean
    euclidean_dis = np.linalg.norm(inp - emb, axis=1)
    emb_ranking = np.argsort(euclidean_dis)
    emb_ranking_distances = euclidean_dis[emb_ranking[:top]]

    emb_ranking_top = emb_ranking[:top]
    euclidean_dis_top = euclidean_dis[emb_ranking_top]

    return emb_ranking_top, euclidean_dis_top

def print_test(model, words, w2i, i2w, epoch):
    model.eval()
    emb_matrix = model.embeddings_target.weight.data.cpu()
    nearest_words_dict = {}

    print('==============================================')
    for w in words:
        inp_emb = emb_matrix[w2i[w], :]

        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top=6)
        with open("../../data/models/summary_guesses.txt",'a',encoding="utf-8") as fp:
            fp.write(f"Epoch: {epoch}:\n{w.ljust(10)} |  {', '.join([i2w[i] for i in emb_ranking_top[1:]])}\n")
        print(w.ljust(10), ' | ', ', '.join([i2w[i] for i in emb_ranking_top[1:]]))
    with open("../../data/models/summary_guesses.txt", 'a', encoding="utf-8") as fp:
        fp.write("\n----------------------------------------------------------------\n")

    return nearest_words_dict