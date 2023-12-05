# -*-coding:utf-8-*-


import pandas as pd
from utils.preprocess import *


train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
print(train_df.corr()['Transported'])


def preprocess(df: pd.DataFrame):
    df.fillna(method='pad', inplace=True)
    df = dummy_process(df, ['HomePlanet', 'Destination'])


    return df


train_df = preprocess(train_df)
test_df = preprocess(test_df)


print(train_df.info())
