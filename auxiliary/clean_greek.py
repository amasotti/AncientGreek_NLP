#-*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'


'''
Prepare texts for tokenizations

'''
import re
import string

def load_text(mypath):
    with open(mypath, 'r',encoding='utf-8') as source:
        text = source.read()
    return text


def clean_greek(text):
    print(f'start cleaning, actual size: {len(text)}')
    # transform ; into ?
    text = re.sub(r';','?',text,0,re.MULTILINE)
    # delete everything which is not text
    text = re.sub(r"\d+(\.\d+)?", r" ", text,0, re.MULTILINE)
    text = re.sub("\d(\.\d\.\d)+", '',text,0,re.MULTILINE)
    text = re.sub(r'\d+', '', text, re.MULTILINE)
    # add whitespace after punctuation
    text = re.sub(r"([.,!?·])", r" \1 ", text,0,re.MULTILINE)
    # delete row break
    text = re.sub(r"\n+", " ", text,0, re.MULTILINE)
    text = re.sub(r"\\n+", " ", text,0, re.MULTILINE)
    text = re.sub(r"n", " ", text,0, re.MULTILINE)
    #delete chapter notation
    text = re.sub(r'ch\s?\.?','',text,re.MULTILINE)
    #delete extra spaces
    text = re.sub(r'\s{2,5}',' ',text,re.MULTILINE)
    text = re.sub(r'  +', ' ', text, 0,re.MULTILINE)
    text = re.sub(r'\s\.\s\.\s', ' . ', text, 0, re.MULTILINE)
    text = re.sub(r'\s·\s.\s',' · ',text,0,re.MULTILINE)
    text = re.sub(r"(\w)(\.|\,|\?|·|;)", "\\1 \\2", data, 0, re.MULTILINE)
    #

    print(f'cleaning successfull; actual size: {len(text)}')
    return text

# apply
'''
text = load_text('../data/Thucydides_raw.txt')
text = clean_greek(text)

print(text[:150])
with open('../data/Thucydides_cleaned.txt','w',encoding='utf-8') as out:
    out.write(text)

'''