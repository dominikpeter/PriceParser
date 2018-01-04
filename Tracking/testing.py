

import pandas as pd
import numpy as np
import csv

df = pd.read_csv("2017-12-08_Price_Tracker.csv", sep=";")

df = df[df['Company']=="Sanitas"]
df = df.fillna('')

df['Key'] = df[['ArtikelId', 'FarbId', 'AusführungsId']].astype(str).apply(lambda x: ''.join(x), axis=1)

pivot = df.pivot(index="Key", columns='Snapshotdate', values='Preis')

df = df.merge(pivot.reset_index(), on="Key")



df = df[['ArtikelId', 'FarbId', 'AusführungsId',
         'Art_Txt_Lang', 'Category_Level_1', 'Category_Level_2',
         'Category_Level_3','2017-11-27', '2017-12-06']].drop_duplicates()

df.to_clipboard()
