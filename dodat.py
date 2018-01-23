import glob
import os
import re
import csv
import time
import datetime

import pandas as pd
import helper as pp


def get_latest_file(path):
    latest_folder = sorted(
        [i for i in os.listdir(path
            ) if re.match('.*Korrekturlauf.*CRH\\b', i)])[0]
    path_ = os.path.join(path, latest_folder)
    return path_


def load_datfile(path):
    files = []
    for file in glob.glob(path+"/**/KOMART0.DAT"):
        if not re.match(".*Getaz*.",file):
            files += [file]

    iter_seq = [6,8,8,1,8,6,20,1,3,1,6,6,6,6,6,1]

    dat = []
    for file in files:
        with open(file) as f:
            for i in f.readlines():
                i_ = 0
                dat_ = []
                for it in iter_seq:
                    dat_ += [i[i_:i_+it].rstrip().lstrip()]
                    i_ += it
                dat += [dat_]
    return dat


def to_df(dat):
    df = pd.DataFrame(dat)
    df.columns = ['Artikelnummer',
                  'Lieferantennummer',
                  'Matchcode',
                  'Mutationscode',
                  'Herstellernummer',
                  'NPK-Nummer',
                  'Konkurrenznummer',
                  'Termincode',
                  'Warengruppe',
                  'Stücklisten-Code',
                  'Länge',
                  'Tiefe',
                  'Höhe',
                  'Breite',
                  'Durchmesser',
                  'Masseinheit']
    df.drop_duplicates(inplace=True)
    return df


def main():
    path_ = os.path.join("\\\\Crhbusadwh02",
                         "SGVSB","Stammdaten")

    pp.create_folder(pp.Path, "Files")
    pp.create_folder(os.path.join(pp.Path, "Files"), "Dotdat")

    now = datetime.datetime.now().strftime("%Y-%m-%d")

    file_path = get_latest_file(path_)
    print("Loading file...\n")
    dat = load_datfile(file_path)
    df = to_df(dat)

    print("Writing csv...\n")
    df.to_csv(os.path.join(
                pp.Path, "Files", "Dotdat",
                now + '_' + 'SGVSB_Dotdat_File.csv'),
                sep=";", index=False, encoding="utf-8")



if __name__ == "__main__":
    print("""\n====================================\n
    Loading Dat File\n
====================================\n""")
    main()
    print("Done...")
