# -*-coding:utf-8-*-


import pandas as pd
from utils.preprocess import *


train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
print(train_df.corr()['Transported'])
consume_list = ['TotalConsume', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def preprocess(df: pd.DataFrame):
    df = dummy_process(df, ['HomePlanet', 'Destination'])
    df.replace({True: 1, False: 0}, inplace=True)
    df.loc[:, 'TotalConsume'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df.loc[df['TotalConsume'] == 0 & df['CryoSleep'].isnull(), 'CryoSleep'] = 1
    df.loc[df['TotalConsume'] != 0 & df['CryoSleep'].isnull(), 'CryoSleep'] = 0
    df.loc[(df['CryoSleep'] == 1) & (df['TotalConsume'].isnull()), consume_list] = 0
    df.loc[df['CryoSleep'] == 0, consume_list].fillna(df.loc[df['CryoSleep'], consume_list].median())



    df.fillna(method='pad', inplace=True)

    return df


train_df = preprocess(train_df)
test_df = preprocess(test_df)

print(train_df.info())
print(train_df['Transported'])
print(train_df.corr()['Transported'])
