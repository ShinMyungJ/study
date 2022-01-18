from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ",len(allAlgorithms))   # 모델의 갯수 :  41(classifier) / 54(regressor)

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

# 모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  [0.89655172 0.86206897 0.85714286 0.85714286 0.96428571]
# BaggingClassifier 의 정답률 :  [0.93103448 1.         0.96428571 0.82142857 1.        ]
# BernoulliNB 의 정답률 :  [0.37931034 0.4137931  0.25       0.25       0.35714286]
# CalibratedClassifierCV 의 정답률 :  [0.96551724 1.         1.         0.89285714 1.        ]
# CategoricalNB 의 정답률 :  [nan nan nan nan nan]
# ClassifierChain 은 에러 터진 놈!!!!
# ComplementNB 의 정답률 :  [0.75862069 0.89655172 1.         0.78571429 0.82142857]
# DecisionTreeClassifier 의 정답률 :  [0.93103448 0.96551724 0.92857143 0.85714286 0.82142857]
# DummyClassifier 의 정답률 :  [0.44827586 0.4137931  0.25       0.5        0.39285714]
# ExtraTreeClassifier 의 정답률 :  [0.82758621 0.86206897 0.85714286 0.85714286 0.89285714]
# ExtraTreesClassifier 의 정답률 :  [0.96551724 1.         1.         0.92857143 1.        ]
# GaussianNB 의 정답률 :  [0.96551724 1.         0.96428571 0.92857143 0.96428571]
# GaussianProcessClassifier 의 정답률 :  [0.96551724 1.         0.96428571 0.89285714 1.        ]
# GradientBoostingClassifier 의 정답률 :  [0.93103448 0.82758621 0.89285714 0.89285714 0.96428571]
# HistGradientBoostingClassifier 의 정답률 :  [0.93103448 1.         0.96428571 0.89285714 1.        ]
# KNeighborsClassifier 의 정답률 :  [0.96551724 1.         1.         0.85714286 1.        ]
# LabelPropagation 의 정답률 :  [0.96551724 0.96551724 0.96428571 0.89285714 0.96428571]
# LabelSpreading 의 정답률 :  [0.96551724 0.96551724 0.96428571 0.89285714 0.96428571]
# LinearDiscriminantAnalysis 의 정답률 :  [1.         0.93103448 0.92857143 0.92857143 1.        ]
# LinearSVC 의 정답률 :  [0.96551724 1.         1.         0.89285714 1.        ]
# LogisticRegression 의 정답률 :  [0.96551724 1.         1.         0.89285714 1.        ]
# LogisticRegressionCV 의 정답률 :  [0.93103448 1.         0.96428571 0.89285714 1.        ]
# MLPClassifier 의 정답률 :  [0.96551724 1.         1.         0.89285714 1.        ]
# MultiOutputClassifier 은 에러 터진 놈!!!!
# MultinomialNB 의 정답률 :  [0.86206897 0.96551724 0.78571429 0.92857143 0.92857143]
# NearestCentroid 의 정답률 :  [0.93103448 1.         0.96428571 0.89285714 1.        ]
# NuSVC 의 정답률 :  [0.96551724 1.         0.96428571 0.92857143 0.96428571]
# OneVsOneClassifier 은 에러 터진 놈!!!!
# OneVsRestClassifier 은 에러 터진 놈!!!!
# OutputCodeClassifier 은 에러 터진 놈!!!!
# PassiveAggressiveClassifier 의 정답률 :  [1.         1.         0.96428571 0.89285714 1.        ]
# Perceptron 의 정답률 :  [0.96551724 1.         0.96428571 0.85714286 1.        ]
# QuadraticDiscriminantAnalysis 의 정답률 :  [0.96551724 0.96551724 0.96428571 1.         1.        ]
# RadiusNeighborsClassifier 의 정답률 :  [0.89655172 0.96551724 0.89285714 0.89285714 0.85714286]
# RandomForestClassifier 의 정답률 :  [0.96551724 1.         1.         0.92857143 1.        ]
# RidgeClassifier 의 정답률 :  [1.         1.         1.         0.89285714 1.        ]
# RidgeClassifierCV 의 정답률 :  [1.         0.96551724 0.96428571 0.89285714 1.        ]
# SGDClassifier 의 정답률 :  [1.         1.         0.96428571 0.89285714 1.        ]
# SVC 의 정답률 :  [1.         1.         1.         0.89285714 1.        ]
# StackingClassifier 은 에러 터진 놈!!!!
# VotingClassifier 은 에러 터진 놈!!!!