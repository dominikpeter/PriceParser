
# coding: utf-8

import argparse
import codecs
import csv
import glob
import math
import json
import os
import re
import datetime
from functools import partial

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

import _main as pp


def create_unique_id(head, tail):
    r = ''
    try:
        if head:
            r = str(head) + ''.join(tail)
    except TypeError:
        r = head
    return r


def save_string_join(x):
    try:
        joined = ''.join(x)
    except TypeError:
        joined = x
    return joined


def check_if_list_or_tuple(object):
    return isinstance(object, (list, tuple))


def check_if_string(object):
    return isinstance(object, str)


def non_or_empty(data, how="all"):
    """ Detect missing values (Nan or Empty)
    ----------
    df : Serie or DataFrame
    how : "all" or "any" """
    switch = {"all": all, "any": any}
    boo = [False for i in np.transpose(data)]
    if isinstance(data, pd.DataFrame):
        boo = pd.isnull(
            data.replace("", np.nan)).apply(
            switch[how.lower()], axis=1)
    if isinstance(data, pd.Series):
        boo = pd.isnull(data.replace("",np.nan))
    return boo


def check_threshold(x, y, threshold):
    x = x.copy()
    y = y.copy()
    x[abs(x - y) > threshold] = np.nan
    return x


def join_and_update(left, right, left_on, right_on,
                    left_update, right_update, joined_on=""):

    assert(check_if_list_or_tuple(
        left_update) == check_if_list_or_tuple(
        right_update))

    if right_on == 'Index':
        right['Index'] = right.index

    if left_on == 'Index':
        left['Index'] = left.index

    if check_if_string(left_update):
        left_update = [left_update]

    if check_if_string(right_update):
        right_update = [right_update]

    left = left.copy()
    righ = right.copy()

    if 'Joined_on' not in left.columns:
        left['Joined_on'] = np.nan

    if check_if_list_or_tuple(left_on):
        for i in left_on:
            left[i] = left[i].replace('', np.nan)
    else:
        left[left_on] = left[left_on].replace('', np.nan)

    left_to_match = left[non_or_empty(left[left_update[0]])]
    left_matched = left[~non_or_empty(left[left_update[0]])]

    mb = left_to_match.shape[0]

    if check_if_list_or_tuple(right_on):
        cols = [i for i in right_on]
    else:
        cols = [right_on]

    for i in right_update:
        cols.append(i)

    right = right[cols].copy()

    if check_if_list_or_tuple(right_on):
        for i in right_on:
            right[i] = right[i].replace('', np.nan)
    else:
        right[right_on] = right[right_on].replace('', np.nan)

    df_join = left_to_match.pipe(save_join, right=right, left_on=left_on,
                                 right_on=right_on,
                                 suffixes=['', '___y'])

    right_update = [i + '___y' for i in right_update]

    check = (non_or_empty(
        df_join[left_update[0]])) & (~non_or_empty(
            df_join[right_update[0]]))

    if check_if_list_or_tuple(joined_on):
        joined_on = " ".join(joined_on)

    df_join.loc[check, 'Joined_on'] = joined_on

    for i, j in zip(left_update, right_update):
        df_join.loc[check, i] = df_join[j]

    df_join = df_join[[i for i in df_join.columns if not i.endswith('___y')]]

    ma = df_join[non_or_empty(df_join[left_update[0]])].shape[0]
    d = mb - ma
    mb += 1.0e-10
    print('\nMatched {} articles from {} ({} %)\n'.format(d, int(round(mb, 0)),
                                                      round(d / mb * 100, 2)))
    left = pd.concat([left_matched, df_join], axis=0).reset_index(drop=True)
    return left


def save_join(left, right, left_on, right_on, *args, **kwargs):
    if right_on == 'Index':
        right['Index'] = right.index
    if left_on == 'Index':
        left['Index'] = left.index

    noe = non_or_empty(right[right_on])

    df_join = left.merge(right[~noe], how="left",
                         left_on=left_on, right_on=right_on,
                         *args, **kwargs)
    return df_join


def prep_dataframe(df):
    df = df.drop_duplicates()
    for i in ['FarbId', 'AusführungsId']:
        df.loc[non_or_empty(df[i]), i] = ''

    df['Farbe'] = df['AF_Txt'].fillna('')
    df['Ausführung'] = df['AFZ_Txt'].fillna('')
    df['FarbId'] = df['FarbId'].replace('', '000')
    df['Preis'] = df['Preis_Pos'].astype(float)
    df['Art_Nr_Hersteller'].astype(str, inplace=True)
    df['Art_Nr_Hersteller'].fillna('', inplace=True)

    df['UID'] = df[['ArtikelId',
                    'FarbId',
                    'AusführungsId']].apply(
        lambda x: create_unique_id(x[0], x[1:]), axis=1)

    df['Art_Nr_Hersteller_UID'] = df[['Art_Nr_Hersteller',
                                      'FarbId',
                                      'AusführungsId']].apply(
        lambda x: create_unique_id(x[0], x[1:]), axis=1)

    if 'Konkurrenzummer' in df.columns:
        df['Konkurrenznummer'].astype(str, inplace=True)
        # df['Konkurrenznummer'].fillna('', inplace=True)
        # df['Konkurrenznummer'] = df[['Konkurrenznummer',
        #                              'FarbId',
        #                              'AusführungsId']].apply(
        #     lambda x: create_unique_id(x[0], x[1:]), axis=1)

    df['Art_Nr_Hersteller'].replace('', np.nan, inplace=True)
    df['Art_Nr_Hersteller_UID'].replace('', np.nan, inplace=True)

    df.loc[df['Art_Nr_Hersteller'].str.len() < 5, 'Art_Nr_Hersteller'] = np.nan
    df.loc[df['Art_Nr_Hersteller_UID'].str.len(
    ) < 5, 'Art_Nr_Hersteller_UID'] = np.nan

    df['EAN'] = df['Preis_EAN'].fillna(df['Art_Nr_EAN'])
    return df


def join_dotdat(df, dotdat, right_on, left_on):
    df = df.pipe(save_join, right=dotdat,
                 right_on=right_on, left_on=left_on,
                 suffixes=['', '___y'])

    def clean_number(x):
        try:
            c = re.sub("\s|\D", "", x)
        except TypeError:
            c = ''
        return c[:6]
    dotdat['Konkurrenznummer'] = dotdat['Konkurrenznummer'].apply(
        lambda x: clean_number(x))
    try:
        df['Konkurrenznummer'] = df['Konkurrenznummer___y']
    except KeyError:
        pass
    df = df[[i for i in df.columns if not i.endswith('___y')]]
    return df


def get_closest(left, right, chunksize=5000,
                threshold=0.5, n_jobs=1, method='cosine',
                columns=['Art_Txt_Lang', 'Art_Txt_Kurz',
                         'Farbe', 'Ausführung']):

    n_jobs = max(1, n_jobs)
    vec = CountVectorizer()

    ix = left.index

    X = vec.fit_transform(left[columns].fillna('').astype(
        str).apply(lambda x: ' '.join(x), axis=1))
    Y = vec.transform(right[columns].fillna('').astype(
        str).apply(lambda x: ' '.join(x), axis=1))

    arr = np.empty((X.shape[0], 2))
    print('Remaining Columns to match = {} ({} Batches)\n'.format(
        X.shape[0], math.ceil(X.shape[0] / chunksize)))

    for i, a in tqdm.tqdm(zip(pp.batch(X, chunksize),
                              pp.batch(arr, chunksize))):
        distance = pairwise_distances(i, Y, metric='cosine', n_jobs=n_jobs)
        distance_min = distance.min(axis=1)
        distance_argmin = distance.argmin(axis=1)
        a[:, 0] = distance_min
        a[:, 1] = distance_argmin

    distance_df = pd.DataFrame(
        arr,
        columns=['Distance', 'Closest'],
        index=ix)

    distance_df['Closest'].astype(np.int, inplace=True)
    distance_df = distance_df[distance_df['Distance'] < threshold]
    return distance_df


def join_prices(left, right, join_keys):
    for i in join_keys:
        jol = ' and '.join(i[0]) if check_if_list_or_tuple(i[0]) else i[0]
        jor = ' and '.join(i[1]) if check_if_list_or_tuple(i[1]) else i[1]

        print('Joining on {} and {}'.format(jol, jor))
        left = left.pipe(join_and_update, right,
                         left_on=i[0], right_on=i[1],
                         left_update=[
                             'Preis_Konkurrenz', 'Txt_Kurz_Konkurrenz',
                             'Txt_Lang_Konkurrenz'],
                         right_update=[
                             'Preis', 'Art_Txt_Kurz', 'Art_Txt_Lang'],
                         joined_on=i[0])
    return left


def join_on_distance(left, right, distance):
    left = left.join(distance, rsuffix='___y')
    left = left.pipe(join_and_update, right,
                     left_on="Closest",
                     right_on='Index',
                     left_update=['Preis_Konkurrenz',
                                  'Txt_Kurz_Konkurrenz', 'Txt_Lang_Konkurrenz'],
                     right_update=['Preis', 'Art_Txt_Kurz', 'Art_Txt_Lang'],
                     joined_on='Text_Similarity')
    return left


def get_price_distance(df):
    if "Preisdifferenz" not in df.columns:
        df['Preisdifferenz'] = np.nan
    df['Preisdifferenz'] = df['Preis'] - df['Preis_Konkurrenz']
    return df


def get_files_dict(rootpath, companies):
    files = os.listdir(rootpath)
    if check_if_string(companies):
        companies = [companies]
    compiler = re.compile(
        r'.*(' + '|'.join(companies).lower() + ')(?!Badmoebel).+')
    files_to_match = [f for f in files if compiler.match(f.lower())]

    files_to_match = {os.path.split(
        i)[-1].split('-')[0]: os.path.join(rootpath, j) for i, j in zip(
        files_to_match, files_to_match)}
    return files_to_match


def get_rank(df):
    assert("Preisdifferenz" in df.columns)
    df['Preisdifferenz_absolut'] = df['Preisdifferenz'].abs()
    rnk = df.sort_values(
        ['UID', 'Preisdifferenz_absolut']).groupby(
        'UID')['Preisdifferenz_absolut'].rank(
        method='first')
    df['Rank'] = rnk
    return df


def delete_with_threshold(df, to_delete, replace=np.nan, threshold=0.5):
    assert("Preisdifferenz" in df.columns)
    assert("Preis" in df.columns)
    check = abs(df['Preisdifferenz'] / df['Preis']) > threshold
    if check_if_list_or_tuple(to_delete):
        for i in to_delte:
            df.loc[check, i] = replace
    else:
        df.loc[check, to_delete] = replace
    return df


def delete_by_rank(df):
    df = df[df['Rank'] == 1]
    return df


def loop_companies(df, dictionary, settings, concat_df=pd.DataFrame()):
    for i in dictionary:
        df_ = df.copy()
        print('\n=================================\n',
            'Preparing {}'.format(i),
            '\n=================================\n'
            )
        compare_df = pp.csv_to_pandas(os.path.join("Output", dictionary[i]))
        df_['Preis_Konkurrenz'] = np.nan
        df_['Txt_Kurz_Konkurrenz'] = np.nan
        df_['Txt_Lang_Konkurrenz'] = np.nan
        df_['Konkurrenz'] = i
        compare_df = compare_df.pipe(prep_dataframe)
        df_ = df_.pipe(join_prices, right=compare_df,
            join_keys=settings['Companies'][i]['Join Keys'])
        distance = df_[pd.isna(
            df_['Preis_Konkurrenz'])].pipe(get_closest,
            compare_df, threshold=0.3)
        df_ = (df_.pipe(join_on_distance, compare_df, distance)
                  .pipe(get_price_distance)
                  .pipe(delete_with_threshold, 'Preis_Konkurrenz')
                  .pipe(get_rank)
                  .pipe(delete_by_rank))
        concat_df = pd.concat([concat_df, df_], axis=0)
    return concat_df


def get_dotdat_file(path, file_):
    dd_files = os.listdir(path)
    dd_files.sort(reverse=True)
    dd_file = os.path.join(path, dd_files[0], file_ + '.csv')
    dotdat = pd.read_csv(dd_file, sep="\t", encoding='utf-8', dtype=str)
    return dotdat


def main(settings):
    rpath = pp.Path
    print("Root path: {}".format(rpath))

    print("Loading main file...")

    gfd = partial(get_files_dict, os.path.join(rpath, "Output"))

    main_file_path = gfd(['Richner'])
    richner = pp.csv_to_pandas(os.path.join(rpath, "Output",
        main_file_path['Richner']))

    print("Preparing main file...")
    richner['Preis_Konkurrenz'] = np.nan
    richner['Konkurrenz'] = 'Sanitas'

    #load dot dat file
    dotdat = get_dotdat_file(
        os.path.join(rpath, "Dotdat", "Output"), "KOMART0")

    richner = richner.pipe(
        join_dotdat, dotdat, left_on="ArtikelId", right_on="Artikelnummer")

    richner = richner.pipe(prep_dataframe)

    companies = gfd([i for i in settings['Companies']])

    final = richner.pipe(loop_companies, companies, settings)

    final = (final[['ArtikelId', 'FarbId', 'AusführungsId', 'UID',
                   'Art_Txt_Kurz', 'Art_Txt_Lang', 'Ausführung', 'Farbe', 'EAN',
                   'Konkurrenz', 'Konkurrenznummer', 'Warengruppe', 'Preis',
                   'Preis_Konkurrenz', 'Txt_Kurz_Konkurrenz',
                   'Txt_Lang_Konkurrenz', 'Joined_on',
                   'Preisdifferenz', 'Art_Nr_Hersteller_Firma',
                   'Category_Level_1', 'Category_Level_2', 'Category_Level_3',
                   'Category_Level_4', 'Closest', 'Distance']]
                   .fillna(''))

    dt = datetime.datetime.now()
    p = os.path.join(rpath, "Matched", dt.strftime("%Y-%m-%d") + "_Output.csv")
    print("Writing File: {}".format(p))
    final.to_csv(p, sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--settings', default="Sanitary", dest="settings",
        help="Name of Setting", type=str)

    args = parser.parse_args()
    setting_to_apply = args.settings

    currentpath = pp.Path
    settings = pp.load_json(os.path.join(
        currentpath, "settings.json"))[setting_to_apply]

    print(
        """{}{}{}Article Matching for Price Comparison
        \n\u00a9 Dominik Peter{}{}{}\n""".format(
            "\n" * 2, "#" * 80, "\n" * 2, "\n" * 2, "#" * 80, "\n" * 2))

    main(settings)
