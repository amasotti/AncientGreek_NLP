# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'

"""
A collection of small scripts to load, clean and extract info from the xml treebank from Perseus

Functions:
    - load_file, load_data
    - extract_verbs : look at the morphological parsing and extracts only the verbs
    - extract_verbs_split : as the one before, but builds two dictionaries: one for active verbs, the other for passive verbs
    - splitted_dictionary : builds a big dictionary with two sub_lists: active and passive
    - extract_forms : given a verb, find the paradigmatic 
    - delete_stopwords 
    - save_load_vocab_json : saves or loads the extracted data
    - print_var : debug function to print some informations while debugging

Classes:
    - Vocabulary : goes through a tokenized text and builds a json /python dictionary
    - Vectorizer : builds vectors (as in the word2vec model) from a vocabulary and a text.


"""

import os
import re
from collections import Counter
import numpy as np
from more_itertools import locate
import json


def load_file(mypath):
    '''
    Read file
    :param mypath: path of the txt file
    :return: string object
    '''
    with open(mypath, 'r', encoding='utf-8') as source:
        testo = source.read()
    return testo


def load_data(data_path):
    '''
        Read file
        :param mypath: path of the txt file
        :return: list object
        '''
    with open(data_path, 'r', encoding='utf-8') as source:
        testo = source.readlines()
    return testo


def extract_verbs(tokens, separator='|'):
    '''
    After having extracted all the infos from the treebanks, this
    small function selects the verbs and returns them in a list
    :param tokens: the list containing the full informations extracted with load_data
    :param separator: the separator used in the txt file (in my case always | )
    :return: a list of verbs
    '''
    verbs = []
    for d in tokens:
        infos = d.split(separator)
        regex = r"v.*"
        match = re.match(regex, infos[2])
        if match is not None:
            verbs.append(infos[1])
    verbs = list(set(verbs))
    try:
        verbs.remove('')
    except:
        print('No empty string in the list, alles ok')
    try:
        verbs.remove(' ')
    except:
        print('No empty string in the list, alles ok')
    return verbs


def extract_verbs_split(tokens, separator='|'):
    '''
    After having extracted all the infos from the treebanks, this
    small function selects the verbs and return two lists, one for the oppositive and one for the mediatantum

    :param tokens: the list containing the full informations extracted with load_data
    :param separator: the separator used in the txt file (in my case always | or , )
    :return: two lists of verbs
    '''
    verbs_oppositive = set()
    verbs_media = set()
    for d in tokens:
        infos = d.split(separator)
        regex = r"v.*"
        match = re.match(regex, infos[2])
        if match is not None:
            regex = r".*μαι"
            match = re.match(regex, infos[1])
            if match is None:
                verbs_oppositive.add(infos[1])
            else:
                verbs_media.add(infos[1])
    try:
        verbs_oppositive.remove('')
        verbs_media.remove('')
    except:
        print('No empty string in the list, alles ok')
    try:
        verbs_oppositive.remove(' ')
        verbs_media.remove(' ')
    except:
        print('No empty string in the list, alles ok')
    return {'mp': list(verbs_media), 'oppositive': list(verbs_oppositive)}


def splitted_dictionary(verb_dictionary, xml_data, labels=['mp', 'oppositive'], save=True, output_fp='data/verb_data_labelled.json'):
    '''
    Creates and optionally saves a json dictionary with v_forms (paradigma) and diathesis
    :param verb_dictionary: a custom dictionary {'mp' : (lst_of_media), 'opp' : (lst of active)}
    :param xml_data: the PoS-Tagging data in list format
    :param labels: per defaul mp, opp. Theoretically applicable also to other splitted dictionaries
    :param save: (bool) save in json format or not
    :return: a dictionary {lemma : { diathesis : 'mp/opp', vforms : [list_paradigm] } }
    '''
    final_dictionary = {}
    for label in labels:
        for verb in verb_dictionary[label]:
            forme = extract_forms(
                verb_list=[verb], data_from_xml=xml_data, save=False)
            final_dictionary[verb] = {
                "diathesis": label, "v_forms": list(forme[verb])}
    if save:
        with open(output_fp, 'w', encoding='utf-8') as fp:
            json.dump(final_dictionary, fp)
    return final_dictionary


def extract_forms(verb_list, data_from_xml, save=False, out_json=None):
    '''
    Given a list of verbs, extracts and optionally saves a list of paradigmatic forms

    :param verb_list: list of verbs (or one verb in list format)
    :param data_from_xml: morphological parsing extracted from treebank
    :param save: bool, save the results
    :param out: path_to_save
    :return: dictionary: verb : (paradigmatic forms)
    '''
    verb_dict = {}
    for verb in verb_list:
        verb_forms = set()
        for d in data_from_xml:
            el = d.split('|')
            if el[1] == verb:
                verb_forms.add(el[0])
        verb_dict[verb] = verb_forms
    if save:
        try:
            with open(out_json, 'w', encoding='utf-8') as jsonDump:
                dump = {lemma: ",".join(
                    _ for _ in verb_dict[lemma]) for lemma in verb_dict.keys()}
                json.dump(dump, jsonDump)
        except:
            print('No output path given')
    return verb_dict


def delete_stopwords(stopwords_file, tokens=None, text=None):
    '''
    Klassische Stopword deletion script

    :param stopwords_file: ein txt datei, jede Zeile ein stopword
    :param tokens: list of tokens to clean
    :param text: alternative to tokens, we can also clean raw texts
    :return: text or token list cleaned
    '''
    with open(stopwords_file, 'r', encoding='utf-8') as stopw:
        stopwords = stopw.read()
    stopwords = stopwords.split("\n")
    if tokens is not None:
        tokens = [w for w in tokens if w not in stopwords]
        return tokens
    if text is not None:
        tokens = text.split()
        tokens = [w for w in tokens if w not in stopwords]
        return tokens


def save_load_vocab_json(out_path, xml_data, verbList):
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as source:
            print('Dictionary already available... Loading it')
            v_forms = json.load(source)
    else:
        print('Dictionary not available... Creating it ex novo')
        v_forms = extract_forms(verb_list=verbList,
                                data_from_xml=xml_data,
                                save=True,
                                out_json=out_path)
    return v_forms


def print_var(**kwargs):
    for k, v in kwargs.items():
        print(f"{k} : {v}")

#################################################################


class Vocabulary():
    def __init__(self, token_to_idx=None, add_unk=None, unk="UNK"):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token,
                             idx in self.token_to_idx.items()}

        self.add_unk = add_unk
        self.unk = unk
        self.unk_idx = -1

        if self.add_unk:
            self.unk_idx = self.add_token(unk)

    def add_token(self, token):
        try:
            idx = self.token_to_idx[token]
        except KeyError:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
        return idx

    def add_many_token(self, token_list):
        return [self.add_token(token) for token in token_list]

    def lookup_token(self, token):
        try:
            if self.unk_idx >= 0:
                return self.token_to_idx.get(token, self.unk_idx)
            else:
                return self.token_to_idx[token]
        except:
            print('Unknown not present and token not in dictionary')

    def lookup_index(self, index):
        if index not in self.idx_to_token.keys():
            raise KeyError('Index not known')
        else:
            return self.idx_to_token[index]

    def to_serial(self):
        return {'token_to_idx': self.token_to_idx,
                'add_unk': self.add_unk,
                'unk': self.unk}

    @classmethod
    def from_serial(cls, content):
        return cls(**content)

    def __str__(self):
        return f'Vocabulary Class for general purpose, len: {len(self)}'

    def __len__(self):
        return len(self.token_to_idx)


############################################################################################################
class Vectorize(object):

    def __init__(self, token_vocab, data_vforms, maxLength_verb_forms=169):
        # eimi hat 169 diverse tokens, so dass man sich an diesem Verb orientieren soll.

        # the vocabulary with all words in the target texts
        self.vocab = token_vocab
        self._vocab_len = len(self.vocab)

        # the dictionray with the paradigmatic forms
        self.data_vforms = data_vforms

    def load_verb_forms(self, verb, separator=","):
        verb_forms = self.data_vforms[verb]
        list_forms = verb_forms.split(separator)
        return list(set(list_forms))

    def find_all_indices(self, token, target):
        indices = list(locate(target, lambda a: a == token))
        return indices

    def find_neighbours(self, target, verb, window=9):
        context = []
        for v_form in self.load_verb_forms(verb):
            indices = self.find_all_indices(v_form, target)
            for occ in indices:
                for j in range(max(occ-window, 0), min(occ+window, len(target))):
                    if j not in indices:
                        context.append(target[j])
        return context

    def extract_common_neighbours(self, target, verb, window=9):
        context = self.find_neighbours(target=target, verb=verb, window=window)
        context = dict(Counter(context))
        return context

    def vectorize_simple(self, verb, target, window=9):
        '''
        Main function for this class
        :param verb: the lemma form to be searched
        :param target: the ordered token list to search for neighbours (ideally,already cleaned from stopwords)
        :param window: window size to search
        :return: a numpy vector of the length of the vocabulary
        '''
        vector = np.zeros(self._vocab_len)
        context = self.extract_common_neighbours(
            target=target, verb=verb, window=window)
        print("LEN CONTEXT AS COUNTER", len(context))
        for token, count in context.items():
            idx = self.vocab.lookup_token(token)
            if not self.vocab.lookup_index(idx) == token:
                break
            vector[idx] = count
        print_var(len_vector_simple=vector.size)
        return vector

    def to_serial(self):
        return {'token_vocab': self.vocab, 'data_vforms': self.data_vforms, 'maxLength_verb_forms': 169}

    def __str__(self):
        return "Vektorizer"
