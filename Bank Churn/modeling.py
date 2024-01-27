# -*-coding:utf-8-*-


import time
import warnings

import xgboost
import catboost
import lightgbm

import numpy as np
import pandas as pd

from sklearn import *

start_t = time.time()
warnings.filterwarnings('ignore')


# read dataset
train, test = pd.read_csv('train_p.csv'), pd.read_csv('test_p.csv')

# feature selection
selection_k_best = feature_selection.SelectKBest(k=35)
selection_k_best.set_output(transform='pandas')
train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
train_Y = train['Exited']
test_X = test[train_X.columns]


s = False
if s:
    # model selection
    models = [catboost.CatBoostClassifier(silent=True), xgboost.XGBClassifier(), lightgbm.LGBMClassifier(verbosity=-1)]

    for K in range(40, 75, 5):
        selection_k_best = feature_selection.SelectKBest(k=K)
        selection_k_best.set_output(transform='pandas')
        train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
        train_Y = train['Exited']
        # ----------------
        CV = 5
        scores = []
        print(K)
        for model in models:
            cross_score = model_selection.cross_val_score(model, train_X, train_Y, cv=CV, scoring='roc_auc')
            scores.append(cross_score)
        print(list(map(lambda x: sum(x) / len(x), scores)))

"""
CV = 5
12
[0.8383104732616113, 0.8792202884836646, 0.8876547999833754, 0.886808047640233, 0.8876468722220103]
14
[0.8490533021433677, 0.8795681062382013, 0.8880734078981348, 0.8868991941225332, 0.887848627092507]
16
[0.8527076335847272, 0.8799995485054165, 0.8886369906950055, 0.8872521448749545, 0.888347865681984]
18
[0.8725611782195253, 0.8802160660783478, 0.8888691155626709, 0.8877606785120589, 0.8883783973625914]
19
[0.8722508993880007, 0.8802783540307051, 0.888905043960347, 0.8882319270577217, 0.8883076480440548]
20
[0.8722086321846515, 0.8802783540307051, 0.8889840109268048, 0.8882319270577217, 0.8883076480440548]
21
[0.8724539128643034, 0.8802783540307051, 0.8888799128070206, 0.8882319270577217, 0.8883075159957103]
22
[0.8721930686671037, 0.8802783540307051, 0.8889032406821038, 0.8882319270577217, 0.8883075159957103]
"""

xgb_p, cat_p, lgb_p = None, None, None
if __name__ == '__main__':
    sample_sub = pd.read_csv('sample_submission.csv')
    xgb, cat, lgb = [True, True, True]
    random_n_iter = 30
    CV = 10
    # ------------------------------------------------------------
    if xgb:
        xgb_param = {
                'eta': np.arange(0.001, 0.1, 0.001),
                'gamma': np.arange(0, 0.2, 0.1),
                'min_child_weight': np.arange(1, 17),
                'max_depth': np.arange(3, 17),
                'n_estimators': np.arange(10, 1000, 10),
                'alpha': np.arange(0, 0.1, 0.02),
                'lambda': np.arange(0, 0.1, 0.02),
                # 'subsample': np.arange(0.70, 1, 0.05),
                # 'colsample_bytree': np.arange(0.70, 1, 0.05)
                }


        model = model_selection.RandomizedSearchCV(xgboost.XGBClassifier(),
                                                   xgb_param, n_iter=random_n_iter, cv=CV, verbose=2)

        model.fit(train_X, train_Y)
        xgb_p = model.best_params_
        # {'subsample': 0.8, 'n_estimators': 100, 'min_child_weight': 7,
        # 'max_depth': 6, 'lambda': 0, 'eta': 0.1, 'alpha': 0.1}

        end_t = time.time()
        cost_time = end_t - start_t
        print(cost_time, '秒')

        predict_proba = model.predict_proba(test_X)
        predict_proba = pd.DataFrame(predict_proba[..., 1], index=sample_sub['id'], columns=['Exited']).astype(float)
        predict_proba.to_csv('xgb_proba.csv')

    # ------------------------------------------------------------
    if cat:
        cat_param = {
                'iterations': np.arange(10, 800, 10),
                'learning_rate': np.arange(0.001, 0.1, 0.001),
                'leaf_estimation_method': ['Newton', 'Gradient'],
                'depth': np.arange(3, 10),
                'boosting_type': ['Ordered', 'Plain'],
                'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
                # 'subsample': np.arange(0.72, 1, 0.02),
                # 'colsample_bylevel': np.arange(0.01, 0.1, 0.02),
                'verbose': [False]
                }

        model = model_selection.RandomizedSearchCV(catboost.CatBoostClassifier(),
                                                   cat_param, n_iter=random_n_iter, cv=CV, verbose=2)

        model.fit(train_X, train_Y)
        cat_p = model.best_params_
        # {'verbose': False, 'one_hot_max_size': 2, 'min_data_in_leaf': 8, 'learning_rate': 1.0, 'l2_leaf_reg': 0.01,
        # 'depth': 4, 'colsample_bylevel': 0.01, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Ordered'}
        # 1870.0954875946045 秒

        end_t = time.time()
        cost_time = end_t - start_t
        print(cost_time, '秒')

        predict_proba = model.predict_proba(test_X)
        predict_proba = pd.DataFrame(predict_proba[..., 1], index=sample_sub['id'], columns=['Exited']).astype(float)
        predict_proba.to_csv('cat_proba.csv')

    # ------------------------------------------------------------
    if lgb:
        lgb_param = {
                'learning_rate': np.arange(0.001, 0.1, 0.001),
                'n_estimators': np.arange(10, 1000, 10),
                'max_depth': np.arange(3, 17),
                'num_leaves': np.arange(100, 1100, 100),
                'num_iterations': np.arange(200, 1000, 200),
                # 'feature_fraction': [0.7, 0.8, 0.9, 1.0],
                # 'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
                # 'reg_alpha': np.arange(0, 500, 1),
                # 'reg_lambda': np.arange(0, 500, 1),
                # 'subsample': np.arange(0.8, 1, 0.02),
                # 'colsample_bytree': np.arange(0.8, 1, 0.02),
                'verbosity': [-1]

                }

        model = model_selection.RandomizedSearchCV(
            lightgbm.LGBMClassifier(callbacks=[lightgbm.log_evaluation(period=1),
                                               lightgbm.early_stopping(stopping_rounds=10)]),
            lgb_param, n_iter=random_n_iter, cv=CV, verbose=2)

        model.fit(train_X, train_Y)
        lgb_p = model.best_params_

        end_t = time.time()
        cost_time = end_t - start_t
        print(cost_time, '秒')


        predict_proba = model.predict_proba(test_X)
        predict_proba = pd.DataFrame(predict_proba[..., 1], index=sample_sub['id'], columns=['Exited']).astype(float)
        predict_proba.to_csv(r'lgb_proba.csv')


    print('xgb best param:', xgb_p)
    print('cat best param:', cat_p)
    print('lgb best param:', lgb_p)

