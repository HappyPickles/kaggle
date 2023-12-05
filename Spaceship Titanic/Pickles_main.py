# -*-coding:utf-8-*-


import pandas as pd
from utils.preprocess import *


train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
print(train_df.corr()['Transported'])


def consume(df):
    return df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']


def preprocess(df: pd.DataFrame):
    df.fillna(method='pad', inplace=True)
    df = dummy_process(df, ['HomePlanet', 'Destination'])
    df.loc[:, 'TotalConsume'] = df.apply(consume, axis=1)
    df.loc[df['TotalConsume'] == 0 & df['CryoSleep'].isnull(), 'CryoSleep'] = 'TRUE'
#    df.loc[df['CryoSleep'] == 'TRUE' & df['TotalConsume'].isnull(), 'TotalConsume', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'] = 0


    return df


train_df = preprocess(train_df)
test_df = preprocess(test_df)

print(train_df.info())
