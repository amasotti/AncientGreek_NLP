#-*- coding: utf-8 -*-

__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'
# Extract from xml

'''
Script to import sentences from Perseus xml datenbank

'''

import xml.etree
from xml.etree import cElementTree as ET
import re

def parse(datei):
    tree = ET.parse(datei)
    root = tree.getroot()
    return root

root = parse('data/thucydides_not_annotiert.xml')
print(root)