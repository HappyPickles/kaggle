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
    df['TotalConsume'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df.loc[(df['TotalConsume'] == 0) & (df['CryoSleep'].isnull()), 'CryoSleep'] = 1
    df.loc[(df['TotalConsume'] != 0) & (df['CryoSleep'].isnull()), 'CryoSleep'] = 0
    df.loc[(df['CryoSleep'] == 1) & (df['TotalConsume'].isnull()), consume_list] = 0
    df.loc[df['CryoSleep'] == 0, consume_list].fillna(df.loc[df['CryoSleep'], consume_list].median())
    df[consume_list] = min_max_norm(df[consume_list])
    df['group_num'] = 0

    cont = 1
    for i in range(df.shape[0]):
        PassengerId = df.loc[i, 'PassengerId']
        group = PassengerId[0:4]
        if i + 1 < df.shape[0]:
            next_PassengerId = df.loc[i+1, 'PassengerId']
            next_group = next_PassengerId[0:4]
            if next_group == group:
                cont += 1
                continue
            else:
                for j in range(cont):
                    df.loc[i-j, 'group_num'] = cont
        else:
            for j in range(cont):
                df.loc[i - j, 'group_num'] = cont
        cont = 1

    df.fillna(method='pad', inplace=True)
    df['deck'] = df['Cabin'].apply(lambda x: x[0])
    df['cabin_num'] = df['Cabin'].apply(lambda x: int(x[2:-2]))
    df['side'] = df['Cabin'].apply(lambda x: x[-1])
    df = dummy_process(df, ['deck', 'side'])

    return df


train_df = preprocess(train_df)
test_df = preprocess(test_df)

print(train_df.info())
print(train_df.corr()['Transported'])
