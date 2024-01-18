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
selection_k_best = feature_selection.SelectKBest(k=20)
selection_k_best.set_output(transform='pandas')
train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
train_Y = train['Exited']
test_X = test[train_X.columns]


s = False
if s:
    # model selection
    models = [catboost.CatBoostClassifier(silent=True), xgboost.XGBClassifier(), lightgbm.LGBMClassifier()]

    for K in [20]:
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
    xgb, cat, lgb = [False, True, False]
    random_n_iter = 30
    CV = 10
    # ------------------------------------------------------------
    if xgb:
        xgb_param = {
                'eta': np.arange(0.001, 1, 0.001),
                'gamma': np.arange(0, 0.2, 0.1),
                'min_child_weight': np.arange(1, 17),
                'max_depth': np.arange(4, 17),
                'n_estimators': np.arange(100, 2100, 100),
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
                'iterations': np.arange(100, 600, 100),
                'learning_rate': np.arange(0.001, 1, 0.001),
                'leaf_estimation_method': ['Newton', 'Gradient'],
                'depth': np.arange(4, 10),
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
                'learning_rate': np.arange(0.0001, 1, 0.0001),
                'n_estimators': np.arange(100, 1000, 100),
                'max_depth': np.arange(4, 17),
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
        predict_proba.to_csv('lgb_proba.csv')


    print('xgb best param:', xgb_p)
    print('cat best param:', cat_p)
    print('lgb best param:', lgb_p)

    """
    28646.701266527176 秒
    
    {'subsample': 1, 'n_estimators': 100, 'min_child_weight': 7, 'max_depth': 6, 'lambda': 0, 'eta': 0.1, 'alpha': 0.1}
    
    {'verbose': False, 'min_data_in_leaf': 8, 'learning_rate': 0.3, 'l2_leaf_reg': 1, 'depth': 4, 
    'colsample_bylevel': 0.1, 'bootstrap_type': 'MVS', 'boosting_type': 'Plain'}
    """

    """
    31747.37921357155 秒
    
    xgb best param: {'subsample': 0.9000000000000001, 'n_estimators': 600, 'min_child_weight': 15, 
    'max_depth': 5, 'lambda': 0.02, 'gamma': 0.1, 'eta': 0.06999999999999999, 
    'colsample_bytree': 0.9500000000000002, 'alpha': 0.02}
    
    cat best param: {'verbose': False, 'subsample': 0.9000000000000001, 'learning_rate': 0.16999999999999998, 
    'leaf_estimation_method': 'Newton', 'iterations': 1200, 'depth': 10, 
    'colsample_bylevel': 0.06999999999999999, 'bootstrap_type': 'MVS', 'boosting_type': 'Ordered'}
    
    lgb best param: {'verbosity': -1, 'subsample': 0.9200000000000002, 'num_leaves': 200, 
    'num_iterations': 1000, 'n_estimators': 900, 'max_depth': 10, 
    'learning_rate': 0.03, 'colsample_bytree': 0.8400000000000001}
    """

    """
    55706.60267615318 秒
    cat best param: {'verbose': False, 'learning_rate': 0.062000000000000006, 'leaf_estimation_method': 'Newton', 
    'iterations': 400, 'depth': 15, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Ordered'}
    
    lgb best param: {'verbosity': -1, 'num_leaves': 600, 'num_iterations': 800, 
    'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.12380000000000001}
    """
