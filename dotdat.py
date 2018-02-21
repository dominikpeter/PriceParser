
import os
import re
import csv
import time
import datetime

import pandas as pd
import numpy as np
import _main as pp


def get_settings(path):
    return pp.load_json(path)

def search_file(path, endswith):
    p = []
    for path_, subdirs, files in os.walk(path):
        for name in files:
            if name.endswith(endswith):
                p.append(os.path.join(path_, name))
    return p


def get_latest_file(path):
    latest_folder = sorted(
        [i for i in os.listdir(path
            ) if re.match('.*Korrekturlauf.*CRH\\b', i)])[0]
    path_ = os.path.join(path, latest_folder)
    return path_


def write_dotdat(df, path, filename):
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    path_ = os.path.join(path, now)
    pp.create_folder(path, now)
    df.to_csv(os.path.join(path_, filename+".csv"),
                sep="\t", index=False, encoding="utf-8")


def load_datfile(files, sequence, columns):
    dat = []
    for file in files:
        with open(file, "r", encoding="ansi") as f:
            try:
                for i in f.readlines():
                    if i:
                        i_ = 0
                        dat_ = []
                        for it in sequence:
                            dat_ += [i[i_:i_+it].rstrip().lstrip()]
                            i_ += it
                        dat += [dat_]
            except UnicodeDecodeError:
                print("Error in Encoding")
                pass
    return to_df(dat, columns)


def to_df(dat, columns):
    df = pd.DataFrame(dat)
    try:
        df.columns = columns
    except ValueError:
        print("Columns do not match")
    df.replace("",np.nan, inplace=True)
    df.replace("\x1a", np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True, how="all")
    for i in df:
        df[i] = df[i].str.replace('', "ü")
        df[i] = df[i].str.replace('”', "ö")
        df[i] = df[i].str.replace('„', "ä")
        df[i] = df[i].str.replace('&', " & ")
    return df


def main():
    path_ = os.path.join("\\\\Crhbusadwh02",
                         "SGVSB","Stammdaten")
    year = str(datetime.datetime.now().year)
    folder = pp.get_sortet_path(
        path_, ".*Korrekturlauf.*"+year+"\sCRH", reverse=True)[0]

    g = [i for i in search_file(
        os.path.join(
        path_, folder), endswith='.DAT') if not re.match(".*Getaz.*", i)]

    settings = pp.load_json(os.path.join(pp.Path, "dotdat.json"))
    print("Schema Version: {}".format(settings['Header']['Version']))

    for dotfile in settings['Files']:
        files = [i for i in g if i.endswith(dotfile+".DAT")]
        df = load_datfile(files, settings['Files'][dotfile]["Sequence"],
            settings['Files'][dotfile]["Columns"])
        print("Writing File: {}".format(dotfile))
        write_dotdat(df, os.path.join(pp.Path, "Dotdat", "Output"), dotfile)



if __name__ == "__main__":
    print("""\n====================================\n
    Loading Dat Files\n
    u00a9 Dominik Peter\n
====================================\n""")
    main()
    print("Done...")
