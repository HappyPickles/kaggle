# \utils\modeling
# -*-coding:utf-8-*-
import pandas as pd

from typing import List
import numpy as np


__all__ = ['ModelParam', 'sorting_average_method']


class ModelParam:

    def __init__(self):

        self.catboost_classifier_params = {
            'iterations': np.arange(200, 2200, 200),
            'learning_rate': np.arange(0.001, 1, 0.001),
            'l2_leaf_reg': np.arange(0.1, 5, 0.1),
            'colsample_bylevel': np.arange(0.001, 0.1, 0.001),
            'leaf_estimation_method': ['Newton', 'Gradient'],
            'depth': np.arange(4, 17),
            'boosting_type': ['Ordered', 'Plain'],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
            'min_data_in_leaf': np.arange(1, 100),
            'subsample': np.arange(0.72, 1.01, 0.02),
            'verbose': [False]
            }

        self.xgboost_classifier_params = {
                'eta': np.arange(0.01, 1, 0.02),
                'gamma': np.arange(0, 0.2, 0.1),
                'min_child_weight': np.arange(1, 17),
                'max_depth': np.arange(4, 17),
                'n_estimators': np.arange(100, 1100, 100),
                'alpha': np.arange(0, 0.2, 0.02),
                'lambda': np.arange(0, 0.2, 0.02),
                'subsample': np.arange(0.70, 1.05, 0.05),
                'colsample_bytree': np.arange(0.70, 1.05, 0.05)
                }

        self.lgtboost_classifier_params = {
            'learning_rate': np.arange(0.001, 1, 0.001),
            'n_estimators': np.arange(100, 1100, 100),
            'max_depth': np.arange(4, 17),
            'num_leaves': np.arange(100, 1100, 100),
            'num_iterations': np.arange(200, 2200, 200),
            'feature_fraction': [0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
            # 'reg_alpha': np.arange(0, 500, 1),
            # 'reg_lambda': np.arange(0, 500, 1),
            'subsample': np.arange(0.72, 1.01, 0.02),
            'colsample_bytree': np.arange(0.72, 1.01, 0.02)
            }


def sorting_average_method(predict_values: np.ndarray):
    """

    :param predict_values:
    :return:
    """
    assert isinstance(predict_values, np.ndarray), \
        TypeError('predict_values must be a list within np.ndarray')


    ret = np.zeros(predict_values.shape[0])

    for i in range(predict_values.shape[1]):
        ret += np.argsort(np.argsort(predict_values[..., i]))

    ret = ret / len(predict_values)
    ret = (ret - ret.min()) / (ret.max() - ret.min())

    return ret


