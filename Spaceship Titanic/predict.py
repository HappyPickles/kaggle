# -*-coding:utf-8-*-


import xgboost as xgb
import pandas as pd
from sklearn import *


train = pd.read_csv(r'train_pre.csv')
test = pd.read_csv(r'test_pre.csv')

features = list(train.corr()['Transported'].sort_values(key=abs).index[::-1])[1:16]


train_X = train.drop(columns=['PassengerId', 'Name', 'Cabin', 'Transported'])[features]
train_Y = train['Transported']
test_X = test.drop(columns=['PassengerId', 'Name', 'Cabin'])[features]

select = False
CV = 10
submit = True
better = True
model_list = [ensemble.RandomForestClassifier(), xgb.XGBClassifier(), ensemble.GradientBoostingClassifier(),
              ensemble.AdaBoostClassifier(), ensemble.BaggingClassifier()]

if __name__ == '__main__':
    model = ensemble.RandomForestClassifier()

    xgb_param = {'eta': [0.01, 0.1, 0.3], 'min_child_weight': [1, 3, 5],
                 'max_depth': [6, 8, 10], 'n_estimators': [100, 500],
                 'alpha': [0, 0.1], 'lambda': [0, 0.2], 'subsample': [0.5, 1]}

    rf_param = {'n_estimators': [100, 200, 500], 'max_depth': [6, 10, 15],
                'max_features': [6, 10], 'bootstrap': [True, False]}

    if select:
        for i, model in enumerate(model_list):
            cross_score = model_selection.cross_val_score(model, train_X, train_Y, cv=CV)
            print(i, sum(cross_score) / CV)

    if better:
        submission_model = model_selection.GridSearchCV(model, param_grid=rf_param)
    else:
        submission_model = model

    if submit:
        submission_model.fit(train_X, train_Y)
        submission = submission_model.predict(test_X)
        submission = pd.DataFrame(submission, index=test['PassengerId'], columns=['Transported'])
        submission['Transported'].replace({1: True, 0: False}, inplace=True)
        submission.to_csv('submission.csv')
        print('Finish')
