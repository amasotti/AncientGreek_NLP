#-*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'
# Extract from xml

'''
Script to import tokens, lemma & POS from Perseus xml datenbank

'''

import xml.etree
from xml.etree import cElementTree as ET
from collections import defaultdict

def parse(datei):
	tree = ET.parse(datei)
	root = tree.getroot()
	return root

root = parse('data/Thucydides_annotiert.xml')
body = root.find('body')

def extract_tokens(body):
	tokens = {}
	for sent in body.findall('sentence'):
		for w in sent.findall('word'):
			if w.attrib != {}:
				#print(w.attrib)
				if 'postag' in w.attrib.keys():
					lemma = ''
					morpho = ''
					token = w.attrib['form']
					if token not in tokens.keys():
						if w.attrib['lemma'] is not None:
							lemma = w.attrib['lemma']
						if w.attrib['postag'] is not None:
							morpho = w.attrib['postag']
						# pos is a list here, to be able to add different parsings
						tokens[token] = {'token':token, 'pos':[morpho], 'lemma':lemma}
					else:
						if morpho in tokens[token]['pos'] or (morpho == '' and len(tokens[token]['pos']) != 0):
							continue
						else:
							tokens[token]['pos'].append(morpho)
	return tokens

tokens = extract_tokens(body= body)


out_path = 'data/Thucydides_data_from_xml.txt'

with open(out_path, 'w',encoding='utf-8') as res:
	for tok, description in tokens.items():
		if description['lemma'] is None:
			lemma = ""
		else:
			lemma = description['lemma']

		if description['pos'] is None:
			pos = ''
		else:
			pos = description['pos']
		final_pos = '|'.join(x for x in pos)
		res.write(tok + "|" + lemma + "|" + final_pos + "\n")

'''
with open(out_path,'w',encoding="UTF-8") as result:
	for tok in tokens:
		form = tok['token']
		lemma = tok['lemma']
		pos = tok['pos']

		if pos == None:
			pos = ""
		if lemma == None:
			lemma = ""
		result.writelines(token + "|" + pos + "|" + lemma + "\n")
'''