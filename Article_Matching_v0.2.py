import codecs
import csv
import glob
import json
import math
import os
import re
import datetime

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import turbodbc


def load_json(path):
    with open(path) as j:
        data = json.load(j)
    return data


def load_sql_text(path):
    with codecs.open(path, encoding='utf-8') as sql:
        file = sql.read()
    return file


def add_columns(df, key):
    colname_preis = 'Preis_{}'.format(key)
    colname_text = 'Txt_Lang_{}'.format(key)
    colname_join = 'Joined_{}_on'.format(key)
    # distance_col = 'Distance_{}'.format(key)
    # closest_col = 'Closest_{}'.format(key)
    cols = [colname_preis,
            colname_text,
            colname_join]

    def check_columns(x):
        if x not in df.columns:
            df[x] = np.nan
    for i in cols:
        check_columns(i)
    return df


def create_connection_string_turbo(server, database):
    options = turbodbc.make_options(prefer_unicode=True)
    constr = 'Driver={ODBC Driver 13 for SQL Server};Server=' + \
        server + ';Database=' + database + ';Trusted_Connection=yes;'
    con = turbodbc.connect(connection_string=constr, turbodbc_options=options)
    return con


def sql_to_pandas(connection, query, *args, **kwargs):
    df = pd.read_sql(query, connection, *args, **kwargs)
    return df


def csv_to_pandas(csv_filepath, *args, **kwargs):
    df = pd.read_csv(csv_filepath, sep=";", dtype=str, *args, **kwargs)
    return df


def check_distance(x, threshold=0.5):
    if len(x) < 2:
        return x
    list_ = x[0]
    i = 1
    list_to_return = [list_]
    while i < len(x):
        r = x[i]
        distance = abs(float(list_) - float(r))
        l_to_check = float(list_) if float(list_) != 0 else np.nan
        if (distance / l_to_check) > threshold:
            r = np.nan
        list_to_return.append(r)
        i += 1
    return list_to_return


def modify_dataframe(df):
    df = df.fillna('')
    df['Farbe'] = df['AF_Txt']
    df['Ausführung'] = df['AFZ_Txt']
    df[['ArtikelId', 'FarbId',
        'AusführungsId',
        'Art_Nr_Hersteller']] = df[['ArtikelId',
                                    'FarbId',
                                    'AusführungsId',
                                    'Art_Nr_Hersteller']].astype(str)
    df['FarbId'] = df['FarbId'].replace('', '000')
    df['SGVSB'] = df[['ArtikelId', 'FarbId', 'AusführungsId']].apply(
        lambda x: ''.join(x), axis=1)
    check_id = df['Art_Nr_Hersteller'].astype(str).apply(lambda x: len(x) > 3)
    df['Art_Nr_Hersteller'] = df['Art_Nr_Hersteller'].replace('', np.nan)

    cols_ = ['FarbId',
             'AusführungsId',
             'Art_Nr_Hersteller',
             'Art_Nr_Hersteller_Firma'
             ]
    df.loc[check_id,
           'idHersteller'] = df.loc[check_id,
                                    cols_].apply(lambda x: ''.join(x), axis=1)
    df['idHersteller'] = df['idHersteller'].replace('\s', '')
    df['EAN'] = df['Preis_EAN'].fillna(df['Art_Nr_EAN'])
    df.iloc[:, 3:] = df.iloc[:, 3:].replace('', np.nan)
    return clean_text(df)


def clean_text(df, pattern='\t|\n|\r'):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = df[i].str.replace(pattern, ' ')
    return df


def batch(iterable, n=1):
    from scipy import sparse
    if sparse.issparse(iterable) or isinstance(
            iterable,
            (np.ndarray, np.generic)):
        row_l = iterable.shape[0]
        for ndx in range(0, row_l, n):
            yield iterable[ndx:min(ndx + n, row_l), ]


def check_input_string_boolean(x):
    if x.lower() in ('yes', 'ja', 'y', 'j', 'true'):
        return True
    if x.lower() in ('no', 'nein', 'n', 'false'):
        return False
    return False


def check_settings(json, key, on):
    c = True
    try:
        c = json[key][on]
    except KeyError:
        print('Key not found \n')
    return c


def join_on_id(df_l, df_r, key, on, settings, threshold=0.5):
    df_r_ = df_r.copy()
    if check_settings(settings, key, on):
        print('Joining Data on {} \n'.format(on))
        try:
            lon, ron = on
        except ValueError:
            lon, ron = on, on
        colname_preis = 'Preis_{}'.format(key)
        colname_text = 'Txt_Lang_{}'.format(key)

        df_l = add_columns(df_l, key)
        df_r_[ron] = df_r_[ron].replace('', np.nan)
        df_l[lon] = df_l[lon].replace('', np.nan)

        df_r_[colname_preis] = df_r_['Preis']
        df_r_[colname_text] = df_r_['Art_Txt_Lang']

        check_ix = df_r_[pd.isnull(df_r_[ron])].index
        df_r_.drop(check_ix, inplace=True)

        df_j = df_l.merge(df_r_, how='left', suffixes=('', '_y'), on=on)

        price = df_j['Preis'].astype(float)
        price_y = df_j['{}_y'.format(colname_preis)].astype(float)

        check_ix_one = df_j[(abs(price - price_y) / price) > threshold].index
        check_ix_two = df_j[pd.notnull(df_j[colname_preis])].index
        df_j.drop(check_ix_one.union(check_ix_two), inplace=True)

        df_j = df_j[[i for i in df_j.columns if re.match('.+_y', i)]]

        df_l = df_l.join(df_j, how='left', lsuffix='', rsuffix='_y')
        df_l = replace_column_after_join(
            df_l, colname_preis, colname_text, key, on)
    else:
        print("Won't join {} on {} due to settings".format(key, on))
    return df_l


def replace_column_after_join(df, colname_preis,
                              colname_text, key, on):
    df['Joined_on_y'] = on
    df[colname_preis] = np.where(
        pd.isnull(df[colname_preis]), df['Preis_y'], df[colname_preis])
    df[colname_text] = np.where(
        pd.isnull(df[colname_text]), df['Art_Txt_Lang_y'], df[colname_text])
    df['Joined_{}_on'.format(key)] = np.where(
        (pd.isnull(df['Joined_{}_on'.format(key)])) & (
            pd.notnull(df['Preis_y'])),
        df['Joined_on_y'], df['Joined_{}_on'.format(key)])
    df = df[[i for i in df.columns if not re.match(
        '.+(_y|Level_5|Level_6)', i)]]
    return df


def join_on_string_distance(df_l, df_r, key,
                            settings, chunksize=5000,
                            threshold=0.5, n_jobs=1, method='cosine',
                            columns=['Art_Txt_Lang', 'Art_Txt_Kurz']):
    df_r_ = df_r.copy()
    on = 'Text Similarity'
    if check_settings(settings, key, on):
        print('Joining Data on {}\n'.format(on))
        df_l = add_columns(df_l, key)
        n_jobs = max(1, n_jobs)
        # vec = TfidfVectorizer(ngram_range=(1, 4))
        vec = CountVectorizer()
        # X_data = main_df[pd.isnull(main_df['Preis Sanitas'])]['Art_Txt_Lang']
        colname_preis = 'Preis_{}'.format(key)
        colname_text = 'Txt_Lang_{}'.format(key)
        # colname_distance = 'Distance_{}'.format(key)

        X = df_l.loc[pd.isnull(df_l[colname_preis]), :]
        ix = X.index
        print('Distance Joining {}\n'.format(key))
        print('Remaining Columns to match = {} ({} Batches)\n'.format(
            X.shape[0], math.ceil(X.shape[0] / chunksize)))

        df_r_[colname_preis] = df_r_['Preis']
        df_r_[colname_text] = df_r_['Art_Txt_Lang']

        X = vec.fit_transform(X[columns].fillna('').astype(
            str).apply(lambda x: ' '.join(x), axis=1))
        Y = vec.transform(df_r_[columns].fillna('').astype(
            str).apply(lambda x: ' '.join(x), axis=1))

        arr = np.empty((X.shape[0], 2))

        for i, a in tqdm.tqdm(zip(batch(X, chunksize), batch(arr, chunksize))):
            distance = pairwise_distances(i, Y, metric='cosine', n_jobs=n_jobs)
            distance_min = distance.min(axis=1)
            distance_argmin = distance.argmin(axis=1)
            a[:, 0] = distance_min
            a[:, 1] = distance_argmin

        colname_distance = 'Distance_{}'.format(key)
        colname_closest = 'Closest_{}'.format(key)

        distance_df = pd.DataFrame(
            arr,
            columns=[colname_distance, colname_closest],
            index=ix)

        begin_x = len(distance_df)
        index_to_drop = distance_df[distance_df['Distance_{}'.format(key)].astype(
            float) > threshold].index
        distance_df.drop(index_to_drop, inplace=True)
        deleted_x = begin_x - len(distance_df)

        print('With a Threshold of {}, deleted {} Rows (% {})\n'.format(
            threshold, deleted_x, round(deleted_x / begin_x * 100, 2)))

        distance_df = distance_df.merge(
            df_r_, how='left', left_on='Closest_{}'.format(key),
            right_index=True, suffixes=('', '_y'))

        df_l = df_l.join(distance_df, how='left', lsuffix='', rsuffix='_y')

        df_l = replace_column_after_join(
            df_l, colname_preis, colname_text, key, on='Text Similarity')
    else:
        print("Won't join {} on {} due to settings\n".format(key, on))
    return df_l


def prepare_data(join_df_path,
                 key, main_df,
                 settings, threshold, distance,
                 n_jobs=0, chunksize=2000):
    join_df = csv_to_pandas(join_df_path)
    join_df = modify_dataframe(join_df)
    main_df = modify_dataframe(main_df)
    main_df = join_on_id(main_df, join_df, key, 'SGVSB',
                         settings['Companies'], threshold=threshold)
    main_df = join_on_id(main_df, join_df, key, 'EAN',
                         settings['Companies'], threshold=threshold)
    main_df = join_on_id(main_df, join_df, key, 'idHersteller',
                         settings['Companies'], threshold=threshold)
    main_df = join_on_string_distance(
        main_df, join_df, key, settings['Companies'],
        threshold=distance, n_jobs=n_jobs, chunksize=chunksize)

    preis_col = [i for i in main_df.loc[:,
                                        'Preis':].columns if re.match('Preis.*', i)]

    main_df[preis_col] = main_df[preis_col].apply(
        lambda x: check_distance(x, threshold), axis=1)
    return main_df


def join_meta_data(main_df, path, sales=True, meta=True):
    con = create_connection_string_turbo('CRHBUSADWH02', 'AnalystCM')
    if sales:
        print('Getting Sales Data from Database...')
        sales_query = load_sql_text(os.path.join(path, 'Sales.sql'))
        sales = sql_to_pandas(con, sales_query)
        main_df = main_df.merge(
            sales, how='left', on='SGVSB',  suffixes=('', '_y'))

    if meta:
        print('Getting Meta Data from Database...')
        meta_query = load_sql_text(os.path.join(path, 'Meta.sql'))
        meta = sql_to_pandas(con, meta_query, parse_dates=['Erstellt_Am'])
        main_df = main_df.merge(
            meta, how='left', on='SGVSB',  suffixes=('', '_y'))
    return main_df


def export_pandas(main_df, path,
                  name='Price-Comparison',
                  to_csv=True, to_excel=True,
                  index=False, timetag=None):
    if timetag:
        filename = os.path.join(path, timetag + '_' + name)
    else:
        filename = os.path.join(path, name)
    try:
        if to_csv:
            filename = filename + '.csv'
            print("Writing CSV File...")
            main_df.to_csv(filename, index=index, sep=';',
                           encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    except PermissionError:
        print('Permission was denied when writing to CSV\n')
    try:
        if to_excel:
            filename = filename + '.xlsx'
            print("Writing Excel File...\n")
            main_df.to_excel(filename,  index=index)
    except PermissionError:
        print('Permission was denied when writing to Excel\n')


def main():
    currentpath = os.getcwd()

    files = [i for i in glob.iglob(os.path.join(
        currentpath) + '/Output/*.csv', recursive=True)]

    settings = load_json(os.path.join(currentpath, 'settings.json'))

    # get settings parameters
    main_file_pattern = settings['Main File']
    text_distance = settings['Max Text Distance']
    price_threshold = settings['Max Price Difference']
    chunksize = settings['Chunksize']
    parallel = settings['Parallel Jobs']
    csv_export = settings['Export']['CSV']
    excel_export = settings['Export']['Excel']
    export_name = settings['Export']['Name']
    timetag_bool = settings['Export']['Timetag']

    timetag = None

    if timetag_bool:
        now = datetime.datetime.now()
        timetag = now.strftime('%Y-%m-%d')

    main_file = [f for f in files if re.match(
        r'.+{}(?!Badmoebel).+'.format(main_file_pattern), f)]

    companies_to_compare = [i for i in settings['Companies']]
    compiler = re.compile(
        r'.+(' + '|'.join(companies_to_compare).lower() + ')(?!Badmoebel).+')
    files_to_match = [f for f in files if compiler.match(f.lower())]
    files_to_match = {os.path.split(
        i)[-1].split('-')[0]: j for i, j in zip(
        files_to_match, files_to_match)}

    main_df = csv_to_pandas(main_file[0])

    for i in files_to_match:
        print('\nMatching Data from {}\n{}\n'.format(i, '#' * 80))
        main_df = prepare_data(
            files_to_match[i], i, main_df,
            settings, threshold=price_threshold,
            distance=text_distance, n_jobs=parallel,
            chunksize=chunksize)
    try:
        main_df = join_meta_data(
            main_df, path=currentpath, sales=True, meta=True)
    except turbodbc.Error:
        print('Cannot connect to Database')

    export_pandas(main_df, path=currentpath, name=export_name,
                  to_csv=csv_export, to_excel=excel_export, timetag=timetag)


# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    print("{}{}{}Article Matching for Price_Comparison\n\n\u00a9 Dominik Peter{}{}{}".format(
        "\n" * 2, "#" * 80, "\n" * 2, "\n" * 2, "#" * 80, "\n" * 2))

    main()
