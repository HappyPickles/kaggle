# -*-coding:utf-8-*-


from utils import modeling

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
sample_sub = pd.read_csv('sample_submission.csv')
train, test = pd.read_csv('train_p.csv'), pd.read_csv('test_p.csv')
train_Y = train['Exited']


# XGB -----------------------------------------------------
selection_k_best = feature_selection.SelectKBest(k=30)
selection_k_best.set_output(transform='pandas')
train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
xgb_test_X = test[train_X.columns]

xgb_model = xgboost.XGBClassifier(n_estimators=1000, min_child_weight=7,
                                  max_depth=6, eta=0.1, alpha=0.01)
xgb_model.fit(train_X, train_Y)
xgb_Y = xgb_model.predict_proba(train_X)
print(xgb_model.score(train_X, train_Y))

# LGB -----------------------------------------------------
selection_k_best = feature_selection.SelectKBest(k=35)
selection_k_best.set_output(transform='pandas')
train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
lgb_test_X = test[train_X.columns]

lgb_model = lightgbm.LGBMClassifier(verbosity=-1, num_leaves=9000, num_iterations=800,
                                    n_estimators=500, max_depth=4, learning_rate=0.05)
lgb_model.fit(train_X, train_Y)
lgb_Y = lgb_model.predict_proba(train_X)
print(lgb_model.score(train_X, train_Y))

# CAT -----------------------------------------------------
selection_k_best = feature_selection.SelectKBest(k=35)
selection_k_best.set_output(transform='pandas')
train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
cat_test_X = test[train_X.columns]

cat_model = catboost.CatBoostClassifier(verbose=False, learning_rate=0.01,
                                        leaf_estimation_method='Gradient', iterations=500, depth=6,
                                        bootstrap_type='Bernoulli', boosting_type='Plain')
cat_model.fit(train_X, train_Y)
cat_Y = cat_model.predict_proba(train_X)
print(cat_model.score(train_X, train_Y))

# Make train_X
proba_arr = np.vstack([xgb_Y[..., 1], lgb_Y[..., 1], cat_Y[..., 1]]).T
fusion_X = pd.DataFrame(proba_arr, columns=['xgb', 'lgb', 'cat']).astype(float)

selection_k_best = feature_selection.SelectKBest(k=30)
selection_k_best.set_output(transform='pandas')
fusion_train_X = selection_k_best.fit_transform(train.drop('Exited', axis=1), train['Exited'])
fusion_test_X = test[train_X.columns]



fusion_X = pd.concat([fusion_X, fusion_train_X], axis=1)
fusion_Y = train_Y.copy()


# fusion
# fusion_model = catboost.CatBoostClassifier()
# fusion_model.fit(fusion_X, fusion_Y)
# print(fusion_model.score(fusion_X, fusion_Y))

test_X_arr = np.vstack([xgb_model.predict_proba(xgb_test_X)[..., 1],
                        lgb_model.predict_proba(lgb_test_X)[..., 1]]).T
# fusion_Y_X = pd.DataFrame(test_X_arr, columns=['xgb', 'lgb', 'cat']).astype(float)
# fusion_test_X = pd.concat([fusion_Y_X, fusion_test_X], axis=1)
#
# predict_proba = fusion_model.predict_proba(fusion_test_X)

predict_proba = modeling.sorting_average_method(test_X_arr)
predict_proba = pd.DataFrame(predict_proba, index=sample_sub['id'], columns=['Exited']).astype(float)

# predict_proba = pd.DataFrame(predict_proba[..., 1], index=sample_sub['id'], columns=['Exited']).astype(float)
predict_proba.to_csv('fusion_proba_1.csv')
