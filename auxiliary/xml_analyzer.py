# -*- coding: utf-8 -*-
__author__ = 'Antonio Masotti'
__date__ = 'Dezember 2020'
# Extract from xml

'''
Script to import tokens, lemma & POS from Perseus xml datenbank

'''

from xml.etree import cElementTree as ET
from auxiliary.GreekToVec_utils import *
import collections
import numpy as np
import pandas as pd
import json
from argparse import Namespace

args = Namespace(
    xml_parsed='../data/vocabularies/Homer_data_from_xml.txt',
    ilias_parsed='../data/treebanks_xml/Ilias.xml',
    odyssee_raw="../data/Odyssee.xml",
    verbs_labelled='../data/verb_data_labelled.json',
    output_json='Ilias_parsed.json',
    output_csv='../data/Media_verbForms.csv',
    extracted_info_csv='../data/Media_HomerGesamt_SubjectsObjects.csv',
    extracted_active_info_csv='../data/Oppositiva_HomerGesamt_SubjectsObjects.csv',
    Homer_Media_counted_functions="../data/Homer_Media_counted_functions.csv",
    Homer_active_counted_functions="../data/Homer_active_counted_functions.csv"
)


def parse(datei):
    """

    :param datei: path to the xml annotated data
    :return: the body element of the xml for further research
    """
    tree = ET.parse(datei).getroot()
    tree_body = tree.find('body')
    return tree_body


def extract_diathesis(data, diathesis='mp'):
    print('Extracting dictionary from json')
    verbs = dict()
    with open(data, 'r', encoding='utf-8') as fp:
        verb_json = fp.read()
        verbs_all = json.loads(verb_json)

    for verb, info in verbs_all.items():
        if info['diathesis'] == diathesis:
            verbs[verb] = info['v_forms']
    print('Verb forms of Verbs in the chosen diathesis extracted')
    return verbs


def save_csv(vocab, output):
    with open(output, 'w', encoding='utf-8') as fp:
        fp.write('verb|forms\n')
        for verb, formList in vocab.items():
            forms = ",".join(_ for _ in formList)
            fp.write(verb + "|" + forms + "\n")
        print(f'Verb forms written to csv in {str(output)}')


def subj_obj_extractor(body, verbs, greedy=False):
    data = []
    # for each media in the list
    for verb, forms in verbs.items():
        single_verb_dictionary = dict()
        single_verb_dictionary['lemma'] = verb
        single_verb_dictionary['subjects'] = list()
        single_verb_dictionary['objects'] = list()
        # for each corresponding verbal form
        for form in forms:
            # seach each sentence in the parsed treebank
            for sentence in body.findall('sentence'):
                # save all the information about this sentence in a dictionary
                infos = dict()
                for word in sentence.findall('word'):
                    current_target = word.attrib['form']
                    infos[current_target] = word.attrib
                    # if the target verb form is in the sentence, proceed with the analysis
                if form in infos.keys():
                    if not greedy:
                        try:
                            head_nr = infos[form]['head']
                            for other_tokens in infos.keys():
                                try:
                                    if infos[other_tokens]['head'] == head_nr and (infos[other_tokens]['relation'] in ['SBJ', 'SBJ_CO']):
                                        single_verb_dictionary['subjects'].append(
                                            infos[other_tokens]['lemma'])
                                    if infos[other_tokens]['head'] == head_nr and (infos[other_tokens]['relation'] in ['OBJ', 'OBJ_CO']):
                                        single_verb_dictionary['objects'].append(
                                            infos[other_tokens]['lemma'])
                                except KeyError:  # if there is no subj, obj continue
                                    continue
                        except:
                            continue
                    elif greedy:
                        for other_tokens in infos.keys():
                            try:
                                if infos[other_tokens]['relation'] in ['SBJ', 'SBJ_CO']:
                                    single_verb_dictionary['subjects'].append(
                                        infos[other_tokens]['lemma'])
                                if infos[other_tokens]['relation'] in ['OBJ', 'OBJ_CO']:
                                    single_verb_dictionary['objects'].append(
                                        infos[other_tokens]['lemma'])
                            except KeyError:  # if there is no subj, obj continue
                                continue
        data.append(single_verb_dictionary)
    return data


def save_extracted_data(dataList, output_fp):
    with open(output_fp, 'w', encoding='utf-8') as fp:
        fp.write('verb|subjects|objects\n')
        for finding in dataList:
            subjects = ','.join(_ for _ in finding['subjects'])
            objects = ','.join(_ for _ in finding['objects'])
            fp.write(finding['lemma'] + "|" + subjects + "|" + objects + "\n")
    print(f'Data written to {str(output_fp)}')


def save(data, output_directory):
    """
    Saves the extracted info in json format
    :param data:
    :param output_directory:
    :return: void
    """
    with open(output_directory, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)
        del data


def extend_findings(preliminary_data, new_source, verb_list, greedy=False):
    print(f'Extending results; Actual length {len(preliminary_data)}')
    body = parse(new_source)
    new_findings = subj_obj_extractor(body, verb_list, greedy=greedy)
    for verb in new_findings:
        lemma = verb['lemma']
        for v in preliminary_data:
            if lemma == v['lemma']:
                if verb['subjects']:
                    v['subjects'].extend(verb['subjects'])
                if verb['objects']:
                    v['objects'].extend(verb['objects'])
    print(f'Results extended; New length {len(preliminary_data)}')
    return preliminary_data


def from_df(data_path):
    data_raw = pd.read_csv(data_path, sep="|", header=0)
    return data_raw


def to_counter(data_df):
    verbs = []
    for _, infoList in data_df.iterrows():
        verb = dict()
        verb['lemma'] = infoList.verb
        verb['subjects'] = []
        verb['objects'] = []
        if not pd.isna(infoList.values[1]):
            verb['subjects'] = collections.Counter(
                infoList.values[1].split(','))
        if not pd.isna(infoList.values[2]):
            verb['objects'] = collections.Counter(
                infoList.values[2].split(','))
        verbs.append(verb)
    return verbs


def save_counter(verbList, out_path):
    with open(out_path, 'w', encoding='utf-8') as fp:
        print('Saving counters to file')
        json.dump(verbList, fp, ensure_ascii=False)


def holistic_overview(counts, save=False, out_path=None):
    subjs = list()
    objs = list()
    for verb in counts:
        subjs.extend(verb['subjects'])
        objs.extend(verb['objects'])
    subjs = collections.Counter(subjs)
    subjs = collections.OrderedDict(subjs.most_common())
    objs = collections.Counter(objs)
    objs = collections.OrderedDict(objs.most_common())
    if save:
        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write('word|function|count\n')
            for k, v in subjs.items():
                fp.write(k + '|subj|' + str(v) + "\n")
            for k, v in objs.items():
                fp.write(k + '|obj|' + str(v) + "\n")
    return subjs, objs


#### IF MAIN  RUN#####
if __name__ == '__main__':
    # extract the xml data
    verbi = load_data(args.xml_parsed)
    # extract the verbs and split after diathesis
    verbs_splitted = extract_verbs_split(verbi)
    # create a splitted dictionary and save to json
    splitted_dictionary(verb_dictionary=verbs_splitted, xml_data=verbi,
                        save=True, output_fp='../data/verb_data_labelled_new.json')

    # extract media from splitted dictionary
    vocab = extract_media(args.verbs_labelled)
    # save verb|v_forms to csv
    save_csv(vocab=vocab, output=args.output_csv)

    #vocabulary = extract_diathesis(args.verbs_labelled,diathesis='mp')
    print('VOCABULARY:')
    print(vocabulary)

    body = parse(args.ilias_parsed)
    subjObjdata = subj_obj_extractor(body, vocabulary, greedy=True)
    subjObjdata = extend_findings(
        preliminary_data=subjObjdata, new_source=args.odyssee_raw, verb_list=vocabulary, greedy=True)
    # Save verbs with their subject object in csv format
    save_extracted_data(subjObjdata, args.extracted_info_csv)
    save_extracted_data(subjObjdata, output_fp=args.extracted_info_csv)

    infos_df = from_df(args.extracted_info_csv)
    print(infos_df.head(25))
    print(infos_df.iloc[3].values[1])
    info_counted = to_counter(infos_df)
    save_counter(info_counted, '../data/HomerGesamt_counted_SbjObj.json')

    # Extract the counts for all media
    sub, obj = holistic_overview(
        info_counted, save=True, out_path=args.Homer_Media_counted_functions)

    # REPEAT FOR THE ACTIVE VERBS FOR COMPARISON
    # Extract actives
'''     vocab = extract_diathesis(data=args.verbs_labelled,
                              diathesis='oppositive')  # 2062 verbs
    body = parse(args.ilias_parsed)
    subjObj_actives = subj_obj_extractor(body=body, verbs=vocab, greedy=True)
    subjObj_actives = extend_findings(
        subjObj_actives, args.odyssee_raw, verb_list=vocab, greedy=True)

    # save actives
    save_extracted_data(
        subjObj_actives, output_fp=args.extracted_gesamt_info_csv)
    info_df_actives = from_df(args.extracted_active_info_csv)
    print(info_df_actives.head(25))
    info_actives_counted = to_counter(info_df_actives)

    subj_act, obj_act = holistic_overview(
        info_actives_counted, save=True, out_path=args.Homer_active_counted_functions) '''
