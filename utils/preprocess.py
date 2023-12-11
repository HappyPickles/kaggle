# \utils\preprocess
# -*-coding:utf-8-*-


from typing import List, Union
import numpy as np
import pandas as pd


__all__ = ['IQR_outlier', 'dummy_process', 'min_max_norm', 'gaussian_norm']


def IQR_outlier(df: Union[pd.DataFrame, np.ndarray],
                features: List[str] = None,
                N: Union[float, int] = 1.5) -> pd.DataFrame:
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
        df = pd.DataFrame(df)
    if features is None:
        features = df.columns
    assert set(features).issubset(set(df.columns)), ValueError('A column does not exist')
    if N <= 0:
        raise ValueError('N must be greater than 0')

    for column in features:
        Q1 = df[column].quantile(0.25)
        Q2 = df[column].quantile(0.75)
        IQR = Q2 - Q1
        df.loc[((df[column] < Q1 - N * IQR) | (df[column] > Q2 + N * IQR)), column] = np.nan

    return df


def dummy_process(df: Union[pd.DataFrame, np.ndarray],
                  features: List[str] = None) -> pd.DataFrame:
    """
    dummy data process.
    :param df:
    :param features:
    :return:
    """

    assert isinstance(df, (pd.DataFrame, np.ndarray)), TypeError('df must be a Pandas DataFrame or NumPy array')
    if type(df) == np.ndarray:
        df = pd.DataFrame(df)

    if features is None:
        features = df.columns
    assert set(features).issubset(set(df.columns)), ValueError('A column does not exist')

    dummies = pd.get_dummies(df[features], prefix_sep='_is_')
    df.drop(features, inplace=True, axis=1)
    df = pd.concat([df, dummies], axis=1)

    return df


def min_max_norm(df: Union[pd.DataFrame, pd.Series]):
    """
    make data scale to 0-1 scale.
    :param df: Data
    :return: norm df.
    """

    assert isinstance(df, (pd.DataFrame, pd.Series)),\
           TypeError('df must be a Pandas DataFrame, Series or NumPy array')

    norm = (df - df.min()) / (df.max() - df.min())

    return norm


def gaussian_norm(df: Union[pd.DataFrame, pd.Series]):
    """
    make data scale to gaussian distribution.
    :param df: Data
    :return: norm df.
    """

    assert isinstance(df, (pd.DataFrame, pd.Series)),\
           TypeError('df must be a Pandas DataFrame, Series or NumPy array')

    norm = (df - df.mean()) / df.std()

    return norm
