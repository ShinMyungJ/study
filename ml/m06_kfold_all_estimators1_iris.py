from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ",len(allAlgorithms))   # 모델의 갯수 :  41(classifier) / 54(regressor)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)
    except:
        # continue
        print(name, '은 에러 터진 놈!!!!')
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# AdaBoostClassifier 의 정답률 :  [0.95833333 0.91666667 0.83333333 0.95833333 0.91666667]
# BaggingClassifier 의 정답률 :  [0.95833333 0.875      0.95833333 0.95833333 0.91666667]
# BernoulliNB 의 정답률 :  [0.29166667 0.375      0.33333333 0.33333333 0.29166667]
# CalibratedClassifierCV 의 정답률 :  [0.91666667 0.91666667 0.83333333 0.79166667 0.875     ]
# CategoricalNB 은 에러 터진 놈!!!!
# ClassifierChain 은 에러 터진 놈!!!!
# ComplementNB 의 정답률 :  [0.70833333 0.625      0.70833333 0.625      0.625     ]
# DecisionTreeClassifier 의 정답률 :  [0.95833333 0.79166667 0.95833333 0.95833333 0.91666667]
# DummyClassifier 의 정답률 :  [0.29166667 0.29166667 0.29166667 0.33333333 0.25      ]
# ExtraTreeClassifier 의 정답률 :  [0.95833333 0.875      0.91666667 0.95833333 0.95833333]
# ExtraTreesClassifier 의 정답률 :  [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667]
# GaussianNB 의 정답률 :  [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667]
# GaussianProcessClassifier 의 정답률 :  [0.95833333 0.91666667 0.875      0.875      0.91666667]
# GradientBoostingClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
# HistGradientBoostingClassifier 의 정답률 :  [0.95833333 0.79166667 0.91666667 0.95833333 0.91666667]
# KNeighborsClassifier 의 정답률 :  [1.         0.91666667 0.95833333 1.         0.91666667]
# LabelPropagation 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
# LabelSpreading 의 정답률 :  [0.95833333 0.95833333 0.91666667 0.95833333 0.91666667]
# LinearDiscriminantAnalysis 의 정답률 :  [1.         1.         0.95833333 0.95833333 1.        ]
# LinearSVC 의 정답률 :  [0.95833333 0.95833333 0.91666667 0.875      0.95833333]
# LogisticRegression 의 정답률 :  [0.95833333 0.91666667 0.91666667 0.875      0.91666667]
# LogisticRegressionCV 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333]
# MLPClassifier 의 정답률 :  [0.95833333 0.91666667 0.95833333 0.91666667 0.95833333]
# MultiOutputClassifier 은 에러 터진 놈!!!!
# MultinomialNB 의 정답률 :  [0.66666667 0.66666667 0.58333333 0.75       0.625     ]
# NearestCentroid 의 정답률 :  [0.95833333 0.91666667 0.95833333 0.91666667 0.875     ]
# NuSVC 의 정답률 :  [0.95833333 1.         0.91666667 0.95833333 0.91666667]
# OneVsOneClassifier 은 에러 터진 놈!!!!
# OneVsRestClassifier 은 에러 터진 놈!!!!
# OutputCodeClassifier 은 에러 터진 놈!!!!
# PassiveAggressiveClassifier 의 정답률 :  [0.83333333 0.95833333 0.70833333 0.79166667 0.83333333]
# Perceptron 의 정답률 :  [0.70833333 0.66666667 0.79166667 0.875      0.66666667]
# QuadraticDiscriminantAnalysis 의 정답률 :  [1.         0.95833333 0.95833333 1.         1.        ]
# RadiusNeighborsClassifier 의 정답률 :  [0.58333333 0.45833333 0.54166667 0.625      0.625     ]
# RandomForestClassifier 의 정답률 :  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
# RidgeClassifier 의 정답률 :  [0.875      0.875      0.83333333 0.79166667 0.79166667]
# RidgeClassifierCV 의 정답률 :  [0.91666667 0.79166667 0.79166667 0.79166667 0.875     ]
# SGDClassifier 의 정답률 :  [0.83333333 0.83333333 0.83333333 0.79166667 0.70833333]
# SVC 의 정답률 :  [0.95833333 0.95833333 0.95833333 1.         0.95833333]
# StackingClassifier 은 에러 터진 놈!!!!
# VotingClassifier 은 에러 터진 놈!!!!