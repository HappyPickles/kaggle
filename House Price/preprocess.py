# /House Price/Preprocess
# -*-coding:utf-8-*-

import pandas as pd
from utils import preprocess as pp

trainDf = pd.read_csv(r'train.csv')
testDf = pd.read_csv(r'test.csv')


continuous_features = []
discontinuous_features = []
target = 'SalePrice'

for df in [trainDf, testDf]:
    pp.IQR_outlier(df)
    pp.dummy_process(df)
