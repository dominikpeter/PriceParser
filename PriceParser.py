import argparse
import codecs
import collections
import csv
import datetime
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import tqdm
import turbodbc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances


def create_folder(path, folder):
    """Folder Creator
    Create Folder if it doesn't exist
    """
    directory = os.path.join(path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_json(path):
    with codecs.open(path, encoding='utf-8') as j:
        data = json.load(j, )

    return data


def load_sql_text(path):
    with codecs.open(path, encoding='utf-8') as sql:
        file = sql.read()

    return file


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


def rec_dd():
    """ Recursive Defaultdict
    Recursive Defaultdict
    """
    return collections.defaultdict(rec_dd)