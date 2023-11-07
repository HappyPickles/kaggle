# -*-coding:utf-8-*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm


# 数据读取
TrainDf = pd.read_csv('train.csv')
TestDf = pd.read_csv('test.csv')
AnsDf = pd.read_csv('gender_submission.csv')

# 数据预处理
for Df in [TrainDf, TestDf]:

    for column in ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        if Df is TestDf and column == 'Survived':
            continue
        Q1 = Df[column].quantile(0.25)
        Q2 = Df[column].quantile(0.75)
        IQR = Q2 - Q1
        N = 1.5
        Df.loc[((Df[column] < Q1 - N * IQR) | (Df[column] > Q2 + N * IQR)), column] = np.nan
        # 超过1.5倍IQR作为异常值删去

    Df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    Df['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    for column in ['Age', 'Fare', 'SibSp', 'Embarked', 'Parch']:
        Df[column].fillna(Df[column].median(), inplace=True)
    Df.loc[Df['Age'] <= 12, 'Age'] = 1
    Df.loc[Df['Age'] > 12, 'Age'] = 0


# 特征选择
# corr = TrainDf.corr()
# print(corr['Survived'])
FeaturesColumns = ['Pclass', 'Sex', 'Age', 'SibSp']
Target = 'Survived'
# print(TrainDf[FeaturesColumns].describe())

# 数据集分割
#
# best = [0, 0, 0]
# for i, model in enumerate([tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(),
#                            neural_network.MLPClassifier(), svm.SVC(), xgb.XGBClassifier()]):
#
#     for size in [0.1 * i for i in range(1, 10)]:
#         S_TrainDf, ValDF = model_selection.train_test_split(TrainDf, test_size=size, random_state=True)
#         model.fit(S_TrainDf[FeaturesColumns], S_TrainDf[Target])
#         val_score = model.score(ValDF[FeaturesColumns], ValDF[Target])
#         test_score = model.score(TestDf[FeaturesColumns], AnsDf[Target])
#         if val_score > best[2]:
#             best[0] = i
#             best[1] = size
#             best[2] = val_score
#         print('当前size为 %.2f ' % size)
#         print('当前model index为 %s' % i)
#         print(val_score)
#         print(test_score)
#         print('--' * 20)
# print(best)



TrainDf, ValDF = model_selection.train_test_split(TrainDf, test_size=0.5, random_state=True)
sub_model = tree.DecisionTreeClassifier()
sub_model.fit(TrainDf[FeaturesColumns], TrainDf[Target])
submission = pd.DataFrame(sub_model.predict(TestDf[FeaturesColumns]),
                          index=TestDf['PassengerId'], columns=['Survived']).astype(int)
print(submission)
submission.to_csv('my_submission.csv')
