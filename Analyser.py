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

import helper as pp


def load_file(filepath):
    df = pp.csv_to_pandas(filepath)

    return df


def modify_dataframe(df):
    ltm_cols = [i for i in df.columns if re.match('.+_LTM', i)]
    df[ltm_cols] = df[ltm_cols].astype(float)

    preis_cols = [i for i in df.columns if re.match('Preis_.*', i)]
    preis_cols = [i for i in preis_cols if i not in ['Preis_EAN']]
    df[preis_cols] = df[preis_cols].astype(float)
    df['Preis'] = df['Preis'].astype(float)
    df['ObjectRate'] = df['ObjectRate'].astype(float)

    df['Erstellt_Am'] = pd.to_datetime(df['Erstellt_Am'])
    df['Avg_Preis'] = df[preis_cols].mean(axis=1)
    df['Std_Preis'] = df[preis_cols].std(axis=1)
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
                'Preis_ArthurWeber','Txt_Lang_ArthurWeber',
                'Joined_ArthurWeber_on',
                'Distance_ArthurWeber','Preis_Briner','Txt_Lang_Briner',
                'Joined_Briner_on','Distance_Briner',
                'Preis_DebrunnerAciferTB',
                'Txt_Lang_DebrunnerAciferTB','Joined_DebrunnerAciferTB_on',
                'Distance_DebrunnerAciferTB',
                'Preis_Engel','Txt_Lang_Engel','Joined_Engel_on',
                'Distance_Engel','Preis_Gabs','Txt_Lang_Gabs',
                'Joined_Gabs_on','Distance_Gabs','Preis_ISOCENTER',
                'Txt_Lang_ISOCENTER','Joined_ISOCENTER_on',
                'Distance_ISOCENTER',
                'Preis_Nussbaum','Txt_Lang_Nussbaum','Joined_Nussbaum_on',
                'Distance_Nussbaum','Preis_Pestalozzi','Txt_Lang_Pestalozzi',
                'Joined_Pestalozzi_on','Distance_Pestalozzi',
                'Preis_SchwarzStahl',
                'Txt_Lang_SchwarzStahl','Joined_SchwarzStahl_on',
                'Distance_SchwarzStahl',
                'Preis_SpaeterHT','Txt_Lang_SpaeterHT','Joined_SpaeterHT_on',
                'Distance_SpaeterHT','Preis_SpaeterHZ','Txt_Lang_SpaeterHZ',
                'Joined_SpaeterHZ_on','Distance_SpaeterHZ','Preis_SpaeterWE',
                'Txt_Lang_SpaeterWE','Joined_SpaeterWE_on',
                'Distance_SpaeterWE',
                'Preis_Tobler','Txt_Lang_Tobler','Joined_Tobler_on',
                'Distance_Tobler','Preis_WalterMeier','Txt_Lang_WalterMeier',
                'Joined_WalterMeier_on','Distance_WalterMeier',
                'GrossSales_LTM','Sales_LTM', 'Margin_LTM',
                'Quantity_LTM', 'ObjectRate',
                'Lieferantenname', 'Erstellt_Am',
                'Artikelserie', 'Wgr-Nr.',
                'Warengruppebezeichnung', 'Preisbasis',
                'Preisfaktor', 'Avg_Preis', 'Std_Preis',
                'Neuer_Preis', 'Neuer_Faktor', 'Neue_Gruppe']

    columns_ = [i for i in columns_ if i in df.columns]

    df = df[columns_]

    timediff_ = datetime.datetime.now() - df['Erstellt_Am']

    select_ = (timediff_.apply(
        lambda x: x.days) < 180) | (
        df['Quantity_LTM'] > 0.0)

    pc_ = ~df['Art_Txt_Lang'].str.contains('ProCasa')

    df = df.loc[select_ & pc_, :].copy().reset_index(drop=True)

    return df


def checks(df):
    mean_std_ = 1
    sales_percentile_ = 75
    quantity_percentile_ = 75
    top_quantity_percentile_ = 95
    top_sales_percentile_ = 95
    object_percentile_ = 40

    c_avg = df['Preis'] < (df['Avg_Preis'] - (df['Std_Preis']) * mean_std_)

    df['Check Avg'] = c_avg
    df['Check_Sales'] = df['Sales_LTM'] < np.nanpercentile(
        df['Sales_LTM'],
        sales_percentile_)

    df['Check_Quantity'] = df[
        'Quantity_LTM'] < np.nanpercentile(
        df['Quantity_LTM'],
        quantity_percentile_)

    high_c = df['Quantity_LTM'] > np.nanpercentile(
        df['Quantity_LTM'],
        top_quantity_percentile_)

    low_s = df['Sales_LTM'] < np.nanpercentile(
        df['Sales_LTM'],
        top_sales_percentile_)

    df['Check_High_Quantity_Low_Sales'] = high_c & low_s

    o_ = df['ObjectRate'] < np.nanpercentile(
        df['ObjectRate'], object_percentile_)

    df['Check_ObjectRate'] = o_

    return df


def write_df(df, output):
    print("Writing file...")
    df.to_csv(output, index=False, sep=';',
              encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)


def main(input_, output_):
    df = load_file(input_)
    df = modify_dataframe(df)
    df_s = select_df(df)
    df_s = checks(df_s)
    df_s = df_s.drop_duplicates()

    write_df(df_s, output_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyser')
    parser.add_argument('-i', default="Price-Comparison.csv",
                        dest="input",
                        help="Input File", type=str)
    parser.add_argument('-o',
                        default="Price-Comparison-Analyser.csv", dest="output",
                        help="Output File", type=str)

    args = parser.parse_args()

    input_ = args.input
    output_ = args.output

    main(input_, output_)
