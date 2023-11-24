# -*-coding:utf-8-*-


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import model_selection

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train)
print(train.head(10))
print(train.tail(10))
print(train.dtypes)
print(train.info())

# 异常值处理

N = 5
upper_quartile = train['Fare'].quantile(0.75)
lower_quartile = train['Fare'].quantile(0.25)
IQR = upper_quartile - lower_quartile
upper_bounds = upper_quartile + N * IQR
lower_bounds = lower_quartile - N * IQR

train['Fare'].hist(bins=50)
plt.show()
train.loc[train['Fare'] > upper_bounds, 'Fare'] = np.nan
train.loc[train['Fare'] < lower_bounds, 'Fare'] = np.nan
train['Fare'].hist(bins=50)
plt.show()

# 缺失值处理

values = {'Fare': train['Fare'].median(),
          'Age': train['Age'].median(),
          'Embarked': train['Embarked'].mode()}
train.fillna(value=values, inplace=True)

train.loc[train['Sex'] == 'female', 'Sex'] = 0
train.loc[train['Sex'] == 'male', 'Sex'] = 1


# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
            'Fare']
target = 'Survived'


#

x_train, x_val, y_train, y_val = model_selection.train_test_split(train[features], train[target], test_size=0.25)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_val, y_val))
