# \utils\preprocess
# -*-coding:utf-8-*-


from typing import List, Dict, Union
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



__all__ = ['percentage_pie']



def percentage_pie(df: Union[pd.DataFrame, np.ndarray],
                   feature: str or List[str], all_in_one: bool = False):
    """

    :param df:
    :param feature:
    :param all_in_one:
    :return:
    """
    assert isinstance(df, (pd.DataFrame, np.ndarray)), \
        TypeError('df must be a Pandas DataFrame or NumPy array')
    if type(df) == np.ndarray:
        df = pd.DataFrame(df)

    if type(feature) == str:
        feature = [feature]
    assert set(feature).issubset(set(df.columns)), \
        ValueError('A column does not exist')

    if all_in_one:
        num_of_feature = len(feature)
        side = int(num_of_feature**0.5)+1
        for fig_i in range(1, num_of_feature+1):
            feature_i = feature[fig_i]
            value_counts = df[feature_i].value_counts()
            plt.subplot(side, side, fig_i)
            plt.pie(value_counts, autopct='%.2f%%')
            plt.legend(value_counts.index)
            plt.title(feature_i)
        plt.show()
    else:
        for feature_i in feature:
            value_counts = df[feature_i].value_counts()
            plt.pie(value_counts, autopct='%.2f%%')
            plt.legend(value_counts.index)
            plt.title(feature_i)
            plt.show()
    return None


def violin(df: Union[pd.DataFrame, np.ndarray],
           features: str or List[str] = None):
    # TODO
    pass


def model_AUC():
    # TODO
    pass





