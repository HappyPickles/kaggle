# -*-coding:utf-8-*-

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier


# 数据读取
TrainDf = pd.read_csv('train.csv')
TestDf = pd.read_csv('test.csv')

# 数据预处理
for Df in [TrainDf, TestDf]:

    for column in ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        if Df is TestDf and column == 'Survived':
            continue
        Q1 = Df[column].quantile(0.25)
        Q2 = Df[column].quantile(0.75)
        IQR = Q2 - Q1
        N = 1.5
        Df.loc[((Df[column] < Q1 - N * IQR) | (Df[column] > Q2 + N * IQR)), column] = np.nan
        # 超过1.5倍IQR作为异常值删去


    Df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    Df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

    for column in ['Age', 'Fare', 'SibSp', 'Embarked']:
        Df[column].fillna(Df[column].median(), inplace=True)



# 特征选择
corr = TrainDf.corr()
FeaturesColumns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
Target = 'Survived'
# print(corr['Survived'])
# print(TrainDf[FeaturesColumns].describe())

# 数据集分割
TrainDf, ValDF = model_selection.train_test_split(TrainDf, test_size=0.25, random_state=True)
model = RandomForestClassifier()
model.fit(TrainDf[FeaturesColumns], TrainDf[Target])
score = model.score(ValDF[FeaturesColumns], ValDF[Target])
print(score)
