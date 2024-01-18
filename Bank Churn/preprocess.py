# -*-coding:utf-8-*-


import warnings
import pandas as pd
import seaborn as sns
from utils.preprocess import *
from matplotlib import pyplot as plt
from sklearn import *

warnings.filterwarnings("ignore")

train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
# --------------------------------------------------


def preprocess(df: pd.DataFrame):
    df = dummy_process(df, feature=['Geography', 'Gender'])
    df['Balance_is_0'] = (df['Balance'] == 0)
    df['Tenure_is_0'] = (df['Tenure'] == 0)
    df['Balance'] = gaussian_norm(df['Balance'])
    df['EstimatedSalary'] = gaussian_norm(df['EstimatedSalary'])
    df = in_range_process(df, {'Age': [[12, 18], [18, 24], [24, 60], [60, 100]],
                               'CreditScore': [[350, 450], [450, 600], [600, 750], [750, 850]]}, drop=False)
    df['Age'] = gaussian_norm(df['Age'])
    df['CreditScore'] = gaussian_norm(df['CreditScore'])
    df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    corr = train_df.corr()
    print(corr['Exited'].sort_values(key=lambda x: abs(x)))

    train_df.to_csv('train_p.csv')
    test_df.to_csv('test_p.csv')
    