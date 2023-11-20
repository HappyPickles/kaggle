# /House Price/Preprocess
# -*-coding:utf-8-*-

import pandas as pd


trainDf = pd.read_csv(r'train.csv')
testDf = pd.read_csv(r'test.csv')


continuous_features = []
discontinuous_features = []
target = 'SalePrice'


