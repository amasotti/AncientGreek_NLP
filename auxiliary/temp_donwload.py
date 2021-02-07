# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'


'''
Script to download and concatenate chapters from Perseus under Philologicus

'''

import requests
import re
import os
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup


def downloadTexts(indirizzo, title):
    '''

    :param indirizzo: url of the work on Perseus under Philologicus
    :param title: title of the work (freie Schnauzte)
    :return: saves the text in a .txt file with utf-8 encoding
    '''
    get_source = requests.get(indirizzo)
    parsed = BeautifulSoup(get_source.text, 'lxml')

    # find all the chapters
    books_urls = re.finditer(
        r"<span class=\"navlevel1\"><a href=\"(.*)\">", str(parsed), re.MULTILINE)

    # find all the titles
    books_titles = re.finditer(
        r"<span class=\"navlevel1\"><a href=\"(.*)\">(.*)<\/a>", str(parsed), re.MULTILINE)

    # create a list with the chapters
    books_link = []
    for link in tqdm(books_urls):
        books_link.append(
            "http://artflsrv02.uchicago.edu/cgi-bin/perseus/"+link.group(1))

    # create a list with the titles (index)
    titles_list = []
    for x in books_titles:
        titles_list.append(x.group(2))

    # save the files
    counter = 0
    for books in books_link:
        print(books)
        get_book = requests.get(books)
        parse_book = BeautifulSoup(get_book.text, 'lxml')
        text = parse_book.find("div", {"id": "perseuscontent"}).text
        file_name = "./Ilias/" + title + "_" + str(titles_list[counter])+".txt"
        with open(file_name, "w+", encoding="UTF-8") as testo:
            testo.write(text)
        counter += 1


homer_address = "http://artflsrv02.uchicago.edu/cgi-bin/perseus/citequery3.pl?dbname=GreekDec20&query=Hom.%20Od.&getid=0"
downloadTexts(homer_address, 'Odyssey')

# Join texts

text_files = glob.glob('./Ilias/*.txt')

with open('Homer_raw.txt', 'w', encoding='utf-8') as out:
    for text in text_files:
        with open(text, 'r', encoding='utf-8') as source:
            content = source.read()
            out.write(content)
