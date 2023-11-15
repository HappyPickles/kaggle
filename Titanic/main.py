# -*-coding:utf-8-*-

import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import *


# Read Datasets
TrainDf = pd.read_csv('train.csv')
TestDf = pd.read_csv('test.csv')

# Preprocess

for Df in [TrainDf, TestDf]:
    for column in ['Survived', 'Pclass', 'Age', 'Fare']:
        if Df is TestDf and column == 'Survived':
            continue
        Q1 = Df[column].quantile(0.25)
        Q2 = Df[column].quantile(0.75)
        IQR = Q2 - Q1
        N = 1.5
        Df.loc[((Df[column] < Q1 - N * IQR) | (Df[column] > Q2 + N * IQR)), column] = np.nan
        # Remove IQR exceeding 1.5 times as outlier.

    Df.loc[((Df['Name'].str.contains('Miss')) & Df['Age'].isna()), 'Age'] = Df.loc[
        ((Df['Name'].str.contains('Miss')) & (Df['Age'].notna())), 'Age'].median()
    Df.loc[((Df['Name'].str.contains('Master')) & Df['Age'].isna()), 'Age'] = Df.loc[
        ((Df['Name'].str.contains('Master')) & (Df['Age'].notna())), 'Age'].median()
    Df.loc[((Df['Name'].str.contains('Mr')) & Df['Age'].isna()), 'Age'] = Df.loc[
        ((Df['Name'].str.contains('Mr')) & (Df['Age'].notna())), 'Age'].median()
    Df.loc[((Df['Name'].str.contains('Mrs')) & Df['Age'].isna()), 'Age'] = Df.loc[
        ((Df['Name'].str.contains('Mrs')) & (Df['Age'].notna())), 'Age'].median()


    Df['Family'] = Df['SibSp'] + Df['Parch']
    Df['Per_Fare'] = Df['Fare'] / (Df['Family'] + 1)


    Df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    Df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    for column in ['Pclass', 'Age', 'Fare', 'Embarked', 'Family', 'Per_Fare', 'SibSp', 'Parch']:
        Df[column].fillna(Df[column].median(), inplace=True)

    Df.loc[Df['Age'] >= 75, 'EaC'] = 2
    Df.loc[Df['Age'] <= 8, 'EaC'] = 1
    Df.loc[((Df['Age'] > 8) & (Df['Age'] < 75)), 'EaC'] = 0

    Df.loc[Df['Cabin'].isna(), 'HC'] = 0
    Df.loc[Df['Cabin'].notna(), 'HC'] = 1

    Df.loc[Df['Cabin'].isna(), 'CabinC'] = 0
    Df.loc[Df['Cabin'].isna(), 'Cabin'] = 'None'
    Df.loc[Df['Cabin'].str.contains(re.compile('[ABC]')), 'CabinC'] = 1
    Df.loc[Df['Cabin'].str.contains(re.compile('[DEF]')), 'CabinC'] = 2
    Df.loc[Df['Cabin'].str.contains(re.compile('[GHT]')), 'CabinC'] = 3

    Df.loc[Df['Ticket'] == 'LINE', 'Ticket'] = '0'
    Df.loc[:, 'TicketN'] = Df['Ticket'].str.extract('(\\d+)', expand=False).astype(int)

    Df.loc[Df['Family'] != 0, 'HF'] = 1
    Df.loc[Df['Family'] == 0, 'HF'] = 0

    Df['TicketN'] = (Df['TicketN'] - Df['TicketN'].min()) / (Df['TicketN'].max() - Df['TicketN'].min())
    Df['Per_Fare'] = (Df['Per_Fare'] - Df['Per_Fare'].min()) / (Df['Per_Fare'].max() - Df['Per_Fare'].min())
    Df['Fare'] = (Df['Fare'] - Df['Fare'].min()) / (Df['Fare'].max() - Df['Fare'].min())


# Features Selection
corr = TrainDf.corr()
print(corr['Survived'])
print('--' * 20)
FeaturesColumns = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Per_Fare', 'EaC', 'HC', 'CabinC', 'TicketN', 'HF']
Target = 'Survived'


# Model Selection
best = [0, 0, 0, 0, 0]
for i, model in enumerate([ensemble.RandomForestClassifier(), xgb.XGBClassifier(),
                           ensemble.GradientBoostingClassifier(), tree.ExtraTreeClassifier(),
                           ensemble.AdaBoostClassifier()]):

    for size in [0.33]:
        S_TrainDf, ValDF = model_selection.train_test_split(TrainDf, test_size=size, random_state=True)
        model.fit(S_TrainDf[FeaturesColumns], S_TrainDf[Target])
        val_score = model.score(ValDF[FeaturesColumns], ValDF[Target])
        train_score = model.score(TrainDf[FeaturesColumns], TrainDf[Target])
        cross_score = model_selection.cross_val_score(model, TrainDf[FeaturesColumns], TrainDf[Target], cv=10)

        if cross_score.mean() > best[4]:
            best[0] = i
            best[1] = size
            best[2] = val_score
            best[3] = train_score
            best[4] = cross_score.mean()
        print('size %.2f ' % size)
        print('model index %s' % i)
        print(val_score)
        print(train_score)
        print(cross_score.mean())
        print('--' * 20)
print(best)


# Predict
# TrainDf, ValDF = model_selection.train_test_split(TrainDf, test_size=0.33, random_state=True)
sub_model = ensemble.RandomForestClassifier()
sub_model.fit(TrainDf[FeaturesColumns], TrainDf[Target])
submission = pd.DataFrame(sub_model.predict(TestDf[FeaturesColumns]),
                          index=TestDf['PassengerId'], columns=['Survived']).astype(int)
submission.to_csv('my_submission.csv')
