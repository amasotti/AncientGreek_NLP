"""
Python script for deaccenting, de-breathning greek texts and deleting stopwords

"""

from argparse import Namespace
from greek_accentuation.characters import strip_accents, strip_breathing
import re

args = Namespace(
    raw_data = "../data/raw_data/Thucydides_raw.txt",
    stopwd_path = "../data/stopwords.txt"
)

# load data
with open(args.raw_data, 'r', encoding='utf-8') as src:
    data = src.read()
    
# load stopwords
with open(args.stopwd_path, 'r', encoding="utf-8") as src:
    stopwords = src.read()
    
stopwords = stopwords.split()

data = re.sub(r"(\d+\.?)+(\w)", " \\2", data, 0, re.MULTILINE)
data = re.sub(r"(\w)(\.|\,|\?|·|;)", "\\1 \\2", data, 0, re.MULTILINE)

# delete stopwords (I know, a kind of gross and clumsy solution)
tokens_raw = data.split()
tokens_raw_cleaned = [w for w in tokens_raw if w not in stopwords]
data = " ".join(w for w in tokens_raw_cleaned)

# remove accents and breathings
data = strip_accents(strip_breathing(data))


# save (I will need this cleaned file for sure later on)
with open("../data/raw_data/ThucydidesGesamt_deaccented.txt", 'w', encoding="utf-8") as fp:
    fp.write(data)