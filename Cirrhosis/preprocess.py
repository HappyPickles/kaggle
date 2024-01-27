# -*-coding:utf-8-*-


import pandas as pd
from utils.data_preprocess import *

train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')


norm_range = {'Bilirubin': [0.3, 1.3], 'Cholesterol': [[0, 200], [200, 240]],
              'Albumin': [[0, 3.5], [3.5, 5.5]], 'Copper': [15, 30], 'Alk_Phos': [45, 130],
              'SGOT': [8, 50], 'Platelets': [100, 300], 'Prothrombin': [11, 13]}


def preprocess(df: pd.DataFrame, test=False):

    df.replace({'Y': 1, 'N': 0}, inplace=True)
    df = dummy_process(df, ['Drug', 'Sex', 'Stage'])
    df = in_range_process(df, norm_range, drop=False)
    df['is_elder'] = 0
    df.loc[df['Age'] >= 365*60, 'is_elder'] = 1
    df.replace({True: 1, False: 0}, inplace=True)

    if not test:
        df = dummy_process(df, ['Status'])

    return df


if __name__ == '__main__':
    pre_train_df = preprocess(train_df)
    pre_test_df = preprocess(test_df, test=True)
    print(pre_train_df.info())
    print(pre_train_df.describe())
    # pre_train_df.to_csv('pre_train.csv')
    # pre_test_df.to_csv('pre_test.csv')
