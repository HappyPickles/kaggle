# -*-coding:utf-8-*-


import xgboost as xgb
import pandas as pd
from sklearn import *


Features = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck', 'HomePlanet_is_Earth', 'HomePlanet_is_Europa', 'TotalConsume',
            'deck_is_B', 'deck_is_C']


train_X = pd.read_csv(r'train_pre.csv')
train_Y = pd.read_csv(r'train_pre.csv')['Transported']
test = pd.read_csv(r'test_pre.csv')
test_X = pd.read_csv(r'test_pre.csv')

train_X.drop(columns=['PassengerId', 'Name', 'Cabin', 'Transported'], inplace=True)
test_X.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
train_X = train_X[Features]
test_X = test_X[Features]

select = True
CV = 10
submit = False


if __name__ == '__main__':
    if select:
        for i, model in enumerate([ensemble.RandomForestClassifier(), xgb.XGBClassifier(),
                                   ensemble.GradientBoostingClassifier(), tree.ExtraTreeClassifier(),
                                   ensemble.AdaBoostClassifier(), tree.DecisionTreeClassifier(),
                                   neural_network.MLPClassifier()]):
            cross_score = model_selection.cross_val_score(model, train_X, train_Y, cv=CV)
            print(i, sum(cross_score) / CV)
    if submit:
        submission_model = ensemble.RandomForestClassifier()
        submission_model.fit(train_X, train_Y)
        submission = submission_model.predict(test_X)
        submission = pd.DataFrame(submission, index=test['PassengerId'], columns=['Transported'])
        submission['Transported'].replace({1: True, 0: False}, inplace=True)
        submission.to_csv('submission.csv')
