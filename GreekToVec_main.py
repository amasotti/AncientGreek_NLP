#-*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'january 2021'

'''
Standford model to reduce texts into Word2Vec Representation

'''

# Imports
from auxiliary.GreekToVec_utils import *
from argparse import Namespace
import os
# Load texts
homer = load_file('data/HomerGesamt_cleaned.txt')


# load infos
with open('data/vocabularies/Homer_data_from_xml.txt', 'r', encoding='utf-8') as source:
    token_data = source.readlines()

# extract all verbs
#verbs = extract_verbs(token_data)
#print("Total verbs in homer: ", len(verbs))

# Organize paths and parameters
args = Namespace(
    homer_text = 'data/HomerGesamt_cleaned.txt',
    thucydides_text = 'data/Thucydides_cleaned.txt',
    data_v_forms = 'data/HomThuc_verb_forms.txt',
    stopwords = 'data/stopwords.txt',
    verbs_homer = None,
    verbs_labelled_homer = None,
    verbForms_homer = 'data/Hom_verb_forms.txt',
    verbForms_homer_json = 'data/Hom_verb_forms.json',
    homer_vocab  = 'data/homer_vocab.json',
    homer_splitted_verbs = 'data/verb_data_labelled.json'
)


# extract verbforms (all, withouth labels)
#v_forms = save_load_vocab_json(out_path=args.verbForms_homer_json, xml_data=token_data, verbList=verbs)
# print(v_forms['δαίω'])
'''
# extract splitted
if not os.path.isfile(args.homer_splitted_verbs):
    print('Dictionary not available, I will create one')
    v_media, v_oppositiva = extract_verbs_split(tokens=token_data)
    splitted_verb = {'mp' : v_media, 'opp': v_oppositiva}
    splitted_verbs = splitted_dictionary(verb_dictionary=splitted_verb, xml_data=token_data,save=True)
else:
    print('Splitted Dictionary available, I will load it')
    with open(args.homer_splitted_verbs,encoding='utf-8') as fp:
        splitted_verbs = json.load(fp)

# Prepare for Deep Learning
coding_labels = {'opp' : 0, 'mp' : 1}

#print(type(splitted_verbs))
#print(splitted_verbs['τίθημι'])
#print("diathese", splitted_verbs['τίθημι']['diathesis'])
#print("Formen von thithemi", splitted_verbs['τίθημι']['v_forms'])
'''

# clean stopwords from Homer and create list of tokens
homer_tokens = delete_stopwords(stopwords_file=args.stopwords,tokens=homer.split())
homer_tok_unique = list(set(delete_stopwords(stopwords_file=args.stopwords,tokens=homer_tokens)))
print("Total tokens", len(homer_tok_unique))
from auxiliary.most_common_words import *

test = create_mostcommon_df(homer_tokens,350,plot=False)
print(test.head(45))

'''
# create vocabulary object
homer_vocab = Vocabulary(add_unk=True)
homer_vocab.add_many_token(homer_tok_unique)

# wenn ein token_to_idx noch nicht gespeichert wurde, tue das
if not os.path.isfile(args.homer_vocab):
    with open('data/homer_vocab.json','w') as fp:
        json.dump(homer_vocab.to_serial(),fp)
    print("Länge Homer_vocabulary: ", len(homer_vocab))

Vectorizer = Vectorize(token_vocab=homer_vocab,
                       data_vforms=v_forms,
                       maxLength_verb_forms=169)

telew_simple = Vectorizer.vectorize_simple(verb='τελέω',target=homer_tokens,window=9)
print_var(size_of_the_vector = telew_simple.size, type_vector = type(telew_simple))

print(Vectorizer.extract_common_neighbours(target=homer_tokens,verb="εἴδω"))

eidw_simple = Vectorizer.vectorize_simple(verb='εἴδω',target=homer_tokens,window=9)
print_var(size_of_the_vector = eidw_simple.size, type_vector = type(eidw_simple))



'''