import argparse
import codecs
import csv
import datetime
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd

import PriceParser as pp


def main(left, right):

    print("Loading lefthandside...")
    main_df = pp.csv_to_pandas(left)

    print("Loading righthandside...")
    join_df = pd.read_excel(right)

    main_df['Join_ArtikelId'] = main_df["ArtikelId"].astype(int)

    main_df = main_df.merge(join_df[['Wgr-Nr.',
                                     'Warengruppebezeichnung',
                                     'Preisbasis',
                                     '02\nRichner\nSoKaFa',
                                     'SGVSB-Nr.']],
                            how="left", left_on="Join_ArtikelId", right_on="SGVSB-Nr.")

    main_df['Preisfaktor'] = main_df['02\nRichner\nSoKaFa']

    main_df = main_df.drop(['Join_ArtikelId',
                            'SGVSB-Nr.',
                            '02\nRichner\nSoKaFa'], axis=1)

    main_df = main_df.drop_duplicates()

    print("Writing file...")
    main_df.to_csv(left, index=False, sep=';',
                   encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--left', default="Price-Comparison.csv",
                        dest="left",
        help="Lefthand File", type=str)
    parser.add_argument('--right',
                        default="Artikel_KorrLauf.xlsx", dest="right",
        help="Righthand File", type=str)

    args = parser.parse_args()

    left = args.left
    right = args.right

    main(left, right)
