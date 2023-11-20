# \utils\preprocess
# -*-coding:utf-8-*-


from typing import List, Union
import numpy as np
import pandas as pd


def IQR_outlier(df: Union[pd.DataFrame, np.ndarray],
                features: List[str] = None,
                N: Union[float, int] = 1.5):
    """
    IQR_outlier. The outlier will be np.nan.
    :param df: DataFrame that need to be processed. If it is Numpy array, it will be transformed to pd.DataFrame.
    :param features: Features that need to be processed, if it is None, then all features will be process.
    :param N: N times. Default to 1.5.
    :return: df with process.
    """

    assert isinstance(N, (float, int)), TypeError('N must be a float or int.')
    assert isinstance(df, (pd.DataFrame, np.ndarray)), TypeError('df must be a Pandas DataFrame or NumPy array')
    if type(df) == np.ndarray:
        Df = pd.DataFrame(df)
    else:
        Df = df.copy()
    if features is None:
        features = Df.columns
    assert set(features).issubset(set(Df.columns)), ValueError('A column does not exist')
    if N <= 0:
        raise ValueError('N must be greater than 0')

    for column in features:
        Q1 = Df[column].quantile(0.25)
        Q2 = Df[column].quantile(0.75)
        IQR = Q2 - Q1
        Df.loc[((Df[column] < Q1 - N * IQR) | (Df[column] > Q2 + N * IQR)), column] = np.nan

    return Df


def dummy_process(df: Union[pd.DataFrame, np.ndarray],
                  features: List[str] = None):
    """
    dummy data process.
    :param df:
    :param features:
    :return:
    """

    assert isinstance(df, (pd.DataFrame, np.ndarray)), TypeError('df must be a Pandas DataFrame or NumPy array')
    if type(df) == np.ndarray:
        Df = pd.DataFrame(df)
    else:
        Df = df.copy()
    if features is None:
        features = Df.columns
    assert set(features).issubset(set(Df.columns)), ValueError('A column does not exist')

    dummies = pd.get_dummies(Df, features)
    Df.drop(features, inplace=True)
    Df.join(dummies)

    return Df
