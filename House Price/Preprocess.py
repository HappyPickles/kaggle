import numpy as np
import pandas as pd


trainDf = pd.read_csv(r'train.csv')
testDf = pd.read_csv(r'test.csv')


continuous_features = []
discontinuous_features = []
target = 'SalePrice'


