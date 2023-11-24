import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import tree

trainDF = pd.read_csv('train.csv')
testDF = pd.read_csv('test.csv')

print(trainDF)
print(trainDF.head(10))
print(trainDF.tail(10))
print(trainDF.dtypes)
print(trainDF.info())


for DF in [trainDF, testDF]:
    for pclass in range(1, 4):

        Fare1 = pd.DataFrame(DF.loc[DF['Pclass'] == Pclass, 'Fare'])
        upper_quartile1 = Fare1.quantile(0.75)

        upper_quartile = DF.loc[DF['Pclass'] == Pclass, 'Fare'].quantile(0.75)
        lower_quartile = DF.loc[DF['Pclass'] == Pclass, 'Fare'].quantile(0.25)

        N = 3
        IQR = upper_quartile - lower_quartile
        upper_bounds = upper_quartile - N * IQR
        lower_bounds = lower_quartile - N * IQR

        DF.loc[DF['Pclass'] == pclass, 'Fare'].loc[DF.loc[DF['Pclass'] == pclass, ] > upper_bounds, ''] = np.nan
        F.loc[F < lower_bounds, F] = np.nan

    values = {'Age': DF['Age'].median(),
              'Fare': DF['Fare'].median(),
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

