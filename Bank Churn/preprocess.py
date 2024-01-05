# -*-coding:utf-8-*-


import warnings
import pandas as pd
import seaborn as sns
from utils.preprocess import *
from matplotlib import pyplot as plt

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
    df['CreditScore'] = gaussian_norm(df['CreditScore'])
    df = in_range_process(df, {'Age': [[12, 18], [18, 24], [24, 60], [60, 100]]}, drop=False)
    df['Age'] = gaussian_norm(df['Age'])
    df.drop(['id', 'CustomerId', 'Surname'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    print(train_df.head().transpose())
    print('--' * 20)
    print(train_df.describe().transpose())
    corr = train_df.corr()
    print(corr)
    sns.heatmap(corr)
    print(corr['Exited'].sort_values(key=lambda x: abs(x)))
    plt.show()
