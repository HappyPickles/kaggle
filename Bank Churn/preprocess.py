# -*-coding:utf-8-*-


import warnings
import pandas as pd
import seaborn as sns
from utils.data_preprocess import *
from matplotlib import pyplot as plt
from sklearn import *

warnings.filterwarnings("ignore")

train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')
# --------------------------------------------------

Q1 = int(test_df['EstimatedSalary'].quantile(0.25))
Q2 = int(test_df['EstimatedSalary'].quantile(0.5))
Q3 = int(test_df['EstimatedSalary'].quantile(0.75))


def preprocess(df: pd.DataFrame):

    CustomerID_counts = df['CustomerId'].value_counts()

    df['CustomerId_count'] = df['CustomerId'].map(lambda x: CustomerID_counts[x])
    df['CustomerId_count'] = min_max_norm(df['CustomerId_count'])
    df['CustomerId_count_is_1'] = (df['CustomerId_count'] == 1)

    df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)

    df = dummy_process(df, feature=['Geography', 'Gender'])
    df.replace({True: 1, False: 0}, inplace=True)

    df = poly_features(df, ['Balance', 'CreditScore', 'Gender_is_Male',
                            'Geography_is_Spain', 'Geography_is_Germany'])

    df['Balance_is_0'] = (df['Balance'] == 0)
    df['Tenure_is_0'] = (df['Tenure'] == 0)
    df['Balance'] = gaussian_norm(df['Balance'])

    df = in_range_process(df, {'Age': [[12, 18], [18, 24], [24, 60], [60, 100]],
                               'CreditScore': [[350, 450], [450, 600], [600, 750], [750, 850]],
                               'EstimatedSalary': [[0, Q1], [Q1, Q2], [Q2, Q3], [Q3, 1000000]]},
                          drop=False)

    df['Age'] = gaussian_norm(df['Age'])
    df['CreditScore'] = gaussian_norm(df['CreditScore'])
    df['EstimatedSalary'] = gaussian_norm(df['EstimatedSalary'])
    df.replace({True: 1, False: 0}, inplace=True)

    return df


if __name__ == '__main__':
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    corr = train_df.corr()
    print(corr['Exited'].sort_values(key=lambda x: abs(x)))

    train_df.to_csv('train_p.csv')
    test_df.to_csv('test_p.csv')
