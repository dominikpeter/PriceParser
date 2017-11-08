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

import Article_Matching as am


def csv_to_pandas(csv_filepath, *args, **kwargs):
    df = pd.read_csv(csv_filepath, sep=";", dtype=str, *args, **kwargs)

    return df


currentpath = os.getcwd()



main_df = am.csv_to_pandas(os.path.join(currentpath,"2017-11-06_Price-Comparison.csv"))
