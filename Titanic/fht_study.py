# -*-coding:utf-8-*-


import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import tree

trainDF = pd.read_csv('train.csv')
testDF = pd.read_csv('test.csv')


for DF in [trainDF, testDF]:
    for pclass in range(1, 4):
        Fare = DF.loc[DF['Pclass'] == pclass, 'Fare']
        upper_quartile = Fare.quantile(0.75)
        lower_quartile = Fare.quantile(0.25)
        N = 3
        IQR = upper_quartile - lower_quartile
        upper_bounds = upper_quartile + N * IQR
        lower_bounds = lower_quartile - N * IQR

        DF.loc[((DF['Pclass'] == pclass) & (DF['Fare'] > upper_bounds)), 'Fare'] = np.nan
        DF.loc[((DF['Pclass'] == pclass) & (DF['Fare'] < lower_bounds)), 'Fare'] = np.nan

        median = DF.loc[DF['Pclass'] == pclass, 'Fare'].median()
        DF.loc[DF['Pclass'] == pclass, 'Fare'].fillna(value=median, inplace=True)

    values = {'Age': DF['Age'].median(),
              'Embarked': DF['Embarked'].mode()}
    DF.fillna(value=values, inplace=True)

    DF.loc[DF['Sex'] == 'female', 'Sex'] = 0
    DF.loc[DF['Sex'] == 'male', 'Sex'] = 1

features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
target = ['Survived']


x_train, x_val, y_train, y_val = model_selection.train_test_split(trainDF[features], trainDF[target], test_size=0.25)
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_val, y_val))

