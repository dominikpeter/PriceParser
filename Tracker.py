import PriceParser as pp
import glob
import os
import re
import pandas as pd
import numpy as np
import csv
import datetime


def search_filetype_in_dict(path, filetype):
    files = [i for i in glob.iglob(path + '/*.{}'.format(str(filetype)),
                                   recursive=True)]
    return files


def concat_dfs(dict_, pattern):
    dict__ = dict_.copy()
    for i in dict__:
        dict__[i] = [i for i in dict__[i]
                     if re.match(".*({}).*".format(pattern), i)]
    df_c = pd.DataFrame()
    for i in dict__:
        df = pp.csv_to_pandas(dict__[i][0])
        cols = ['ArtikelId',
                'FarbId',
                'AusführungsId',
                'Art_Txt_Lang',
                'Preis',
                'Category_Level_1',
                'Category_Level_2',
                'Category_Level_3',
                'Category_Level_4']
        df = df[cols]
        df['Snapshotdate'] = i
        df['Snapshotdate'] = df['Snapshotdate'].astype('datetime64[ns]')

        df_c = pd.concat([df_c, df], axis=0)

    return df_c


def gather_data(dict_, patterns):
    df_c = pd.DataFrame()
    for i in patterns:
        try:
            df = concat_dfs(dict_, i)
            df['Snapshotdate_String'] = df['Snapshotdate'].apply(
                lambda x: 'Snapshotdate ({})'.format(
                    str(x.strftime("%Y-%m-%d"))))

            df['Company'] = str(i)
            df = df.fillna('')
            df_c = pd.concat([df_c, df], axis=0)
        except (KeyError, IndexError):
            print("Pattern {} not found...".format(i))
            pass
    return df_c


def main(currentpath, pattern):
    now = datetime.datetime.now()

    folders = os.listdir(os.path.join(currentpath, "Archiv"))
    folders.sort()

    d = {}
    for i in folders:
        d[i] = search_filetype_in_dict(
            os.path.join(currenpath, "Archiv", i), "csv")

    df = gather_data(d, pattern)

    path_to_save = os.path.join(currentpath,
        'Tracking', '{}_Price_Tracker.csv'.format(now.strftime("%Y-%m-%d")))

    df.to_csv(path_to_save, index=False,
              sep=';', encoding='utf-8',
              quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--settings', default="Sanitary", dest="settings",
        help="Name of Setting", type=str)

    args = parser.parse_args()
    setting_to_apply = args.settings

    currentpath = pp.Variables.Path
    # currentpath = os.getcwd()

    settings = pp.load_json(os.path.join(currentpath, 'settings.json'))
    settings = settings[setting_to_apply]

    pattern = [i for i in settings['Companies']]

    pp.create_folder(currenpath, "Tracking")

    main(currentpath, pattern)
