

import pandas as pd
import numpy as np
import json
import os
import random
import PriceParser as pp

path = pp.Folderpath.Path

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    try:
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    except TypeError:
        return ['background-color: white']

def load_file(filepath):
    df = pp.csv_to_pandas(filepath)

    return df

c = os.getcwd()
path_ = os.path.join(c, "Matching", "2017-11-13_Price-Comparison-Sanitary-Analysis.csv")

df = load_file(path_)

def highlight_cols(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #select all values to default value - red color
    df.loc[:,:] = 'background-color: white'
    df.loc[df['Joined_Sanitas_on'] == 'Text Similarity','Preis_Sanitas'] = 'background-color: yellow'
    #overwrite values grey color
    #return color df
    return df

df.style.apply(highlight_cols, axis=None).to_excel('test.xlsx')
