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

def load_file(filepath):
    df = pp.csv_to_pandas(filepath)

    return df


def modify_dataframe(df):
    ltm_cols = [i for i in df.columns if re.match('.+_LTM', i)]
    df[ltm_cols] = df[ltm_cols].astype(float)

    preis_cols = [i for i in df.columns if re.match('Preis.*', i)]
    df[preis_cols] = df[preis_cols].astype(float)
    df['ObjectRate'] = df['ObjectRate'].astype(float)

    df['Erstellt_Am'] = pd.to_datetime(df['Erstellt_Am'])
    df['Avg_Preis'] = np.mean(df[preis_cols], axis=1)
    df['Std_Preis'] = np.std(df[preis_cols], axis=1)
    df['Neuer_Preis'] = np.nan
    df['Neuer_Faktor'] = np.nan
    df['Neue_Gruppe'] = np.nan

    return df


def select_df(df):
    columns_ = ['ArtikelId', 'FarbId', 'AusführungsId',
                'Art_Nr_Hersteller', 'Art_Nr_Hersteller_Firma',
                'Art_Txt_Lang', 'Preis',
                'Category_Level_1', 'Category_Level_2',
                'Preis_Sabag', 'Txt_Lang_Sabag',
                'Joined_Sabag_on', 'Distance_Sabag',
                'Preis_Saneo', 'Txt_Lang_Saneo',
                'Joined_Saneo_on', 'Distance_Saneo',
                'Preis_Sanitas', 'Txt_Lang_Sanitas',
                'Joined_Sanitas_on', 'Distance_Sanitas',
                'Preis_TeamHug', 'Txt_Lang_TeamHug',
                'Joined_TeamHug_on', 'Distance_TeamHug',
                'Preis_TeamSaniDusch', 'Txt_Lang_TeamSaniDusch',
                'Joined_TeamSaniDusch_on', 'Distance_TeamSaniDusch',
                'GrossSales LTM','Sales_LTM', 'Margin_LTM',
                'Quantity_LTM', 'ObjectRate',
                'Lieferantenname', 'Erstellt_Am',
                'Artikelserie', 'Wgr-Nr.',
                'Warengruppebezeichnung', 'Preisbasis',
                'Preisfaktor']
    df = df[columns_]

    timediff_ = datetime.datetime.now() - df['Erstellt_Am']

    select_  = (timediff_.apply(
                    lambda x: x.days) < 180) | (
                df['Quantity_LTM'] > 0.0)

    pc_ = ~df['Art_Txt_Lang'].str.contains('ProCasa')

    df_select = df.loc[select_ & pc_, :].copy().reset_index(drop=True)

    return df_select

def checks(df):
    mean_std_ = 1
    sales_percentile_ = 75
    quantity_percentile_ = 75
    top_quantity_percentile_ = 95
    top_sales_percentile_ = 95
    object_percentile_ = 40

    c_avg = df_select['Preis'] < (df_select[
            'Avg_Preis']-(df_select['Std_Preis'])*mean_std_)

    df_select['Check Avg'] = c_avg
    df_select['Check_Sales'] = df_select[
                                        'Sales_LTM'] < np.nanpercentile(
                                            df_select['Sales_LTM'],
                                                sales_percentile_)

    df_select['Check_Quantity'] = df_select[
                                        'Quantity_LTM'] < np.nanpercentile(
                                            df_select['Quantity_LTM'],
                                                quantity_percentile_)

    high_c = df_select[
                'Quantity_LTM'] > np.nanpercentile(
                                    df_select[
                                        'Quantity_LTM'],
                                            top_quantity_percentile_)

    low_s = df_select['Sales_LTM'] < np.nanpercentile(
                                            df_select[
                                             'Sales_LTM'],
                                                top_sales_percentile_)

    df_select[
        'Check_High_Quantity_Low_Sales'] = high_c & low_s


    o_ = df_select[
            'ObjectRate'] < np.nanpercentile(
                                df_select['ObjectRate'], object_percentile_)

    df_select['Check_ObjectRate'] = o_

    return df_select

def write_df(df, output):
    print("Writing file...")
    df.to_csv(output, index=False, sep=';',
                   encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


def main(input_, output_):
    df = load_file(input_)
    df = modify_dataframe(df)
    df_s = select_df(df)
    write_df(df_s, output_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example with long option names')
    parser.add_argument('--input', default="Price-Comparison.csv",
                        dest="input",
        help="Input File", type=str)
    parser.add_argument('--output',
                        default="Price-Comparison-Analyser.csv", dest="output",
        help="Output File", type=str)

    args = parser.parse_args()

    input_ = args.input
    output_ = args.output

    main(input_, output_)
