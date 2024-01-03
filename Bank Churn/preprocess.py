# -*-coding:utf-8-*-


import pandas as pd
from utils.preprocess import *


train_df = pd.read_csv(r'train.csv')
test_df = pd.read_csv(r'test.csv')

print(train_df.info())
print(train_df.describe())



def preprocess(df: pd.DataFrame):

    pass


