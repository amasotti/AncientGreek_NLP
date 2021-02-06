# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'


'''
Prepare texts for tokenizations

'''
import re
from argparse import Namespace
from greek_accentuation.characters import strip_accents, strip_breathing
import re

args = Namespace(
    # raw text, as downloaded from Perseus
    raw_data="../data/raw_data/Thucydides_raw.txt",
    stopwd_path="../data/stopwords.txt",  # stopword list for greek
    cleaned_text="../data/raw_data/cleaned_text.txt"
)


def load_text(mypath):
    """
    Dummy function to just load text from disk

    """
    with open(mypath, 'r', encoding='utf-8') as source:
        text = source.read()
    return text


def clean_greek(text):
    """
    Common cleaning tasks with regex for greek texts

    """
    print(f'start cleaning, actual size: {len(text)}')
    # transform ; into ?
    text = re.sub(r';', '?', text, 0, re.MULTILINE)
    # delete everything which is not text
    text = re.sub(r"\d+(\.\d+)?", r" ", text, 0, re.MULTILINE)
    text = re.sub("\d(\.\d\.\d)+", '', text, 0, re.MULTILINE)
    text = re.sub(r'\d+', '', text, re.MULTILINE)
    # add whitespace after punctuation
    text = re.sub(r"([.,!?·])", r" \1 ", text, 0, re.MULTILINE)
    # delete row break
    text = re.sub(r"\n+", " ", text, 0, re.MULTILINE)
    text = re.sub(r"\\n+", " ", text, 0, re.MULTILINE)
    text = re.sub(r"n", " ", text, 0, re.MULTILINE)
    # delete chapter notation
    text = re.sub(r'ch\s?\.?', '', text, re.MULTILINE)
    # delete extra spaces
    text = re.sub(r'\s{2,5}', ' ', text, re.MULTILINE)
    text = re.sub(r'  +', ' ', text, 0, re.MULTILINE)
    text = re.sub(r'\s\.\s\.\s', ' . ', text, 0, re.MULTILINE)
    text = re.sub(r'\s·\s.\s', ' · ', text, 0, re.MULTILINE)
    text = re.sub(r"(\w)(\.|\,|\?|·|;)", "\\1 \\2", data, 0, re.MULTILINE)
    #

    print(f'cleaning successfull; actual size: {len(text)}')
    return text


# apply -- Loads and cleans the text
if __name__ == '__main__':
    # Load text
    text = load_text(args.raw_data)
    # apply cleaning
    text = clean_greek(text)

    # load stopwords
    with open(args.stopwd_path, 'r', encoding="utf-8") as src:
        stopwords = src.read()

    # Delete stopwords (I know, a kind of gross and clumsy solution)
    tokens_raw = text.split()
    tokens_raw_cleaned = [w for w in tokens_raw if w not in stopwords]
    data = " ".join(w for w in tokens_raw_cleaned)

    # remove accents and breathings
    data = strip_accents(strip_breathing(data))

    print(data[:150])
    with open(args.cleaned_text, 'w', encoding='utf-8') as out:
        out.write(data)
