import argparse
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
import turbodbc

import PriceParser as pp

currenpath = os.getcwd()


main_df = pp.csv_to_pandas(os.path.join(currenpath,
                              "Matching",
                              '2017-11-10_Price-Comparison-Sanitary.csv'))


mean_std = 1
sales_percentile = 0.75
quantity_percentile = 0.75
object_percentile = 0.4


main_df.columns
# [(i,j) for i, j in zip(range(len(main_df.columns)), main_df.columns)]

main_df['Neuer Preis'] = np.nan
main_df['Neuer Faktor'] = np.nan
main_df['Neue Gruppe'] = np.nan
main_df['Avg Preis'] = np.mean(main_df[preis_cols], axis=1)
main_df['Std Preis'] = np.std(main_df[preis_cols], axis=1)



columns = []
[columns.append(i) for i in main_df.columns[:5]];
[columns.append(i) for i in main_df.columns[23:26]];
[columns.append(i) for i in main_df.columns[29:]];

preis_cols = [i for i in main_df.columns if re.match('Preis_.+', i)]
main_df[preis_cols] = main_df[preis_cols].astype(float)

main_df_select = main_df[columns].copy().reset_index(drop=True)














if __name__ == '__main__':
    main()
