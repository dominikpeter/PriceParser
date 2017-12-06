# -*- coding: utf-8 -*-
"""XML Parser

Parse XML

"""

import argparse
import sys
import csv
import glob
import multiprocessing
import os
import re
from functools import partial
import datetime


import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm
import PriceParser as pp


def switch_to_right(c):
    if any(c):  # at least one have to be a value
        i = 0
        while i < len(c):
            try:
                if not c[i]:
                    c[i] = c[i + 1]
                    c[i + 1] = None
            except IndexError:
                pass
            i += 1
        try:
            if not c[0]:
                switch_to_right(c)
        except IndexError:
            pass
    return c


def switch_to_left(c):
    if any(c):  # at least one have to be a value
        i = 0
        while i < len(c):
            try:
                if not c[i]:
                    c[i] = c[i - 1]
            except IndexError:
                pass
            i += 1
    return c


def get_nodes(element, k=[]):
    if element.attrib != {}:
        try:
            k += [element.attrib['Txt']]
        except:
            pass
        get_nodes(element.getparent(), k)
    return k


def get_tree(file):
    """ XML Tree
    Get XML Tree
    """
    try:
        parser = etree.XMLParser()
        tree = etree.parse(file, parser=parser)
        return tree
    except TypeError:
        print('Error')


def array_to_string(array, sep=' '):
    if isinstance(array, list):
        return sep.join(array)
    return array


def findall_loop(element, tag, attrib=None):
    x = ''
    if attrib:
        try:
            x = [i.attrib[attrib] for i in element.findall(tag)]
        except KeyError:
            pass
    else:
        try:
            x = [i.text for i in element.findall(tag)]
        except KeyError:
            pass
    if x:
        return x


class XML_Parser:
    """XML Parser Class

    """

    def __init__(self, tree):
        self.tree = tree
        self.dict_ = pp.rec_dd()
        self.cat_dict = {}

        self.attr_dict = {
            "Art_Nr_Anbieter": '',
            "Art_Nr_Hersteller": '',
            "Art_Nr_Hersteller_Firma": '',
            "Art_Nr_EAN": '',
            "Art_Nr_Nachfolge": '',
            "Art_Nr_Synonym": '',
            "Art_Nr_Synonym_Firma": '',
            "Art_Valid_Von": '',
            "Art_Valid_Bis": '',
            "Art_Txt_Kurz": '',
            "Art_Txt_Lang": '',
            "Art_Menge": '',
            "BM_Einheit_Code": '',
            "BM_Einheit_Code_BM_Einheit": '',
            "Preis_Pos": '',
            "Preis_EAN": '',
            "AF_Nr": '',
            "AF_Txt": '',
            "AFZ_Txt": '',
            "AFZ_Nr": ''
        }

        self.DF = pd.DataFrame()
        self.catDF = pd.DataFrame()

    def update_dict(self, level1, level2, level3, key, value):
        try:
            level1 = array_to_string(level1)
            level2 = array_to_string(level2)
            level3 = array_to_string(level3)
            try:
                value = array_to_string(value)
            except (IndexError, KeyError, TypeError):
                value = str(value)
        except (IndexError, KeyError, TypeError):
            print(KeyError)

        self.dict_[str(level1)][str(level2)][str(
            level3)][str(key)] = str(value)

    def save_attrib(self, element, attr):
        try:
            return element.attrib[attr]
        except KeyError:
            return ''

    def clear_variables(self):
        for i in self.attr_dict:
            self.attr_dict[i] = ''

    def insert_dict(self, level1, level2, level3):
        for i in self.attr_dict:
            self.update_dict(level1, level2, level3, i, self.attr_dict[i])

    def parse_xml(self):
        for artikel in self.tree.findall('.//Artikelmenge/Artikel'):
            self.clear_variables()
            try:

                for i in ['Art_Nr_Hersteller_Firma', 'Art_Nr_Synonym']:
                    self.attr_dict[i] = findall_loop(
                        artikel, './/{}'.format(i), 'Firma')

                self.attr_dict["BM_Einheit_Code_BM_Einheit"] = findall_loop(
                    artikel, 'BM_Einheit', './/BM_Einheit_Code')

                self.attr_dict["Art_Nr_Anbieter"] = artikel.attrib[
                    'Art_Nr_Anbieter']
                # print(self.attr_dict["Art_Nr_Anbieter"])

                attr_to_loop = ['Art_Nr_Hersteller',
                                'Art_Nr_EAN',
                                'Art_Nr_Nachfolge',
                                'Art_Nr_Synonym',
                                'Art_Nr_Synonym_Firma',
                                'Art_Valid_Von',
                                'Art_Valid_Bis',
                                'Art_Txt_Kurz',
                                'Art_Menge',
                                'BM_Einheit_Code']

                for i in attr_to_loop:
                    self.attr_dict[i] = findall_loop(
                        artikel, './/{}'.format(i))

                preisaf = artikel.findall('.//Preis_AF')
                preiszu = artikel.findall('.//Preis_AF_Zusatz')

                if not preisaf and not preiszu:
                    try:
                        for preis in artikel.findall('.//Preis'):

                            for i in ['Preis_Pos', 'Preis_EAN']:
                                self.attr_dict[i] = findall_loop(
                                    preis, './/{}'.format(i))

                            self.insert_dict(
                                self.attr_dict["Art_Nr_Anbieter"], '', '')

                    except KeyError:
                        print("Error")
                        pass

                elif preisaf and not preiszu:
                    try:
                        for preisaf in preisaf:
                            for af in preisaf.findall('.//AF'):
                                self.attr_dict["Preis_Pos"] = findall_loop(
                                    af, './/Preis_Pos')
                                self.attr_dict["Preis_EAN"] = findall_loop(
                                    af, './/EAN')
                                self.attr_dict["AF_Nr"] = findall_loop(
                                    af, './/AF_Nr')
                                self.attr_dict["AF_Txt"] = findall_loop(
                                    af, './/AF_Txt')

                                self.insert_dict(
                                    self.attr_dict["Art_Nr_Anbieter"],
                                    self.attr_dict["AF_Nr"], '')
                    except KeyError:
                        print("Error")
                        pass

                elif preiszu and not preisaf:
                    try:
                        for preiszu in preiszu:
                            for afz in preiszu.findall('.//AFZ'):
                                try:
                                    self.attr_dict["AF_Nr"] = findall_loop(
                                        afz, './/AF_Nr')
                                    self.attr_dict["AF_Txt"] = findall_loop(
                                        afz, './/AF_Txt')
                                    for afznr in afz.findall('.//AFZ_Nr'):
                                        self.attr_dict["Preis_Pos"] = self.save_attrib(
                                            afznr, 'Preis')
                                        self.attr_dict["Preis_EAN"] = self.save_attrib(
                                            afznr, 'EAN')
                                        self.attr_dict["AFZ_Nr"] = afznr.text
                                        self.attr_dict["AFZ_Txt"] = self.save_attrib(
                                            afznr, 'Txt')
                                        self.insert_dict(
                                            self.attr_dict["Art_Nr_Anbieter"],
                                            self.attr_dict["AF_Nr"],
                                            self.attr_dict["AFZ_Nr"])
                                except KeyError as e:
                                    print('Error', e)
                                    pass
                    except KeyError as e:
                        print('Error', e)
            except KeyError:
                pass

    def get_category_to_dict(self):
        d = {}
        for i in self.tree.xpath('/DataExpert/Body/Katalog/Suchbegriffe/Register_Suche/Register_Element_1'):
            for n in range(10):
                for j in i.findall('.//Element{}_Nr'.format(n)):
                    d[j.text] = j
        for i in d:
            k = []
            self.cat_dict[i] = get_nodes(d[i], k)

    def category_dict_to_df(self):
        self.catDF = pd.DataFrame.from_dict(self.cat_dict, orient='index')
        self.catDF.columns = ['Category_Level_{}'.format(
            str(len(self.catDF.columns) - i)) for i in range(len(self.catDF.columns))]
        self.catDF = self.catDF[['Category_Level_{}'.format(
            str(i + 1)) for i in range(len(self.catDF.columns))]]
        self.catDF = self.catDF.apply(lambda l: switch_to_right(l), axis=1)
        self.catDF = self.catDF.apply(lambda l: switch_to_left(l), axis=1)
        self.catDF['ArtikelId'] = self.catDF.index

    def dict_to_df(self, filename=''):
        if self.dict_:
            self.DF = pd.DataFrame.from_dict({(i, j, k): self.dict_[i][j][k]
                                              for i in self.dict_.keys()
                                              for j in self.dict_[i].keys()
                                              for k in self.dict_[i][j].keys()},
                                             orient='index')

            self.DF['ArtikelId'] = [i[0] for i in self.DF.index]
            self.DF['FarbId'] = [i[1] for i in self.DF.index]
            self.DF['AusführungsId'] = [i[2] for i in self.DF.index]
            self.DF['xml'] = str(filename)
            self.DF['Preis'] = self.DF['Preis_Pos']

            try:
                self.DF = self.DF[['ArtikelId',
                                   'FarbId',
                                   'AusführungsId',
                                   'Preis_Pos',
                                   'Preis_EAN',
                                   'Art_Nr_Hersteller',
                                   'Art_Nr_Hersteller_Firma',
                                   'Art_Nr_EAN',
                                   'Art_Nr_Nachfolge',
                                   'Art_Nr_Synonym',
                                   'Art_Nr_Synonym_Firma',
                                   'Art_Valid_Von',
                                   'Art_Valid_Bis',
                                   'Art_Txt_Kurz',
                                   'Art_Txt_Lang',
                                   'Art_Menge',
                                   'BM_Einheit_Code',
                                   'BM_Einheit_Code_BM_Einheit',
                                   'AF_Nr',
                                   'AF_Txt',
                                   'AFZ_Txt',
                                   'AFZ_Nr',
                                   'Preis']]

                self.DF['Preis_Pos'] = pd.to_numeric(
                    self.DF['Preis_Pos'], errors='coerce')
            except KeyError as err:
                print('Error converting the DataFrame. Error: ', err)

            self.DF = self.DF.replace(to_replace=np.nan, value='')
            self.DF = self.DF.replace(to_replace='[None]', value=np.nan)
            self.DF = self.DF.replace(to_replace='None', value=np.nan)

    def create_filenames(self, filename='', sep='-'):
        filename = os.path.split(
            filename)[-1].replace('.xml', '').replace('.XML', '').replace('.csv', '')
        try:
            filename_split = filename.split(
                '-')[0] + sep + filename.split('-')[1]
        except IndexError:
            filename_split = filename
            print(filename, filename_split)
        return (filename, filename_split)

    def merge_categories(self):
        self.get_category_to_dict()
        self.category_dict_to_df()
        self.DF = self.DF.merge(self.catDF, how='left',
                                on='ArtikelId', suffixes=('', ''))

    def df_to_file(self, filename, path, csv_, archiv_, excel_):
        if not self.DF.empty:

            xlsx_ext = '.xlsx'
            csv_ext = '.csv'

            filename, filename_split = self.create_filenames(filename)
            if csv_:
                path_to_save = os.path.join(
                    path, 'Output', filename_split + csv_ext)
                self.DF.to_csv(path_to_save, index=False,
                               sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
                if archiv_:
                    now = datetime.datetime.now().strftime('%Y-%m-%d')
                    path_to_save = os.path.join(
                        path, 'Archiv', now, filename + csv_ext)
                    self.DF.to_csv(path_to_save, index=False,
                                   sep=';', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
            if excel_:
                path_to_save = os.path.join(
                    path, 'Excel', filename_split + xlsx_ext)
                self.DF.to_excel(path_to_save,  index=False)


def process_xml(path__, archiv__, csv__, excel__, file__):
    tree = get_tree(file__)
    Parser = XML_Parser(tree)
    Parser.parse_xml()
    Parser.dict_to_df(os.path.split(file__)[-1])
    Parser.merge_categories()
    Parser.df_to_file(filename=file__, path=path__,
                      archiv_=archiv__, csv_=csv__, excel_=excel__)
    #print('Processed: Filename {}'.format(file__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--n_jobs', default=0, dest="n_jobs",
                        help="Number of Parallel Jobs", type=int)
    parser.add_argument('--pattern', default=None, dest="pattern",
                        help="Pattern to match XMLs (comma separated)", type=str)
    parser.add_argument('--archiv', default='yes',
                        dest="archiv", help="Archiv CSV's", type=str)
    parser.add_argument('--csv', default='yes', dest="csv_arg",
                        help="Generate CSV Files", type=str)
    parser.add_argument('--excel', default='no', dest="excel_arg",
                        help="Generate Excel Files", type=str)

    currentpath = pp.Path

    args = parser.parse_args()

    if not os.path.exists(os.path.join(currentpath, 'XML')):
        print('Put the XML Document in a Subfolder called XML')
        exit()

    __, XMLS = pp.get_xmls(currentpath)

    n_jobs = args.n_jobs
    archiv = pp.check_input_string_boolean(args.archiv)
    excel_arg = pp.check_input_string_boolean(args.excel_arg)
    csv_arg = pp.check_input_string_boolean(args.csv_arg)

    if archiv:
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        pp.create_folder(currentpath, os.path.join("Archiv", now))
    if excel_arg:
        pp.create_folder(currentpath, "Excel")
    if csv_arg:
        pp.create_folder(currentpath, "Output")

    if args.pattern:
        MATCH = args.pattern.split(",")
        REC = compiler = re.compile(r'.+(' + '|'.join(MATCH).lower() + ')')
        XMLS = [i for i in XMLS if compiler.search(i.lower())]

    print("\n\n\n==============================================\n\n",
          "XML Parser",
          "\n",
          "Parsing XML Files",
          "\n\n""==============================================\n\n\n")

    for xml in XMLS:
        print("Processing file: {}".format(xml))

    print("\n")

    if excel_arg and n_jobs > 0:
        print('Export Excel in Parallel is not recommended\n\nProcessing files...\n\n')
    else:
        print('Processing files...\n\n')

    if n_jobs > 0:
        func = partial(process_xml, currentpath, archiv, csv_arg, excel_arg)
        pool = multiprocessing.Pool(n_jobs)
        pool.map(func, XMLS)
    else:
        for xml in XMLS:
            xml_text = os.path.split(xml)[-1]
            print("Parsing {}{}".format(xml_text, "\t" * 5), end="\r")
            process_xml(currentpath, archiv, csv_arg, excel_arg, xml)
