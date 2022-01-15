from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
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
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        continue
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# AdaBoostClassifier 의 정답률 :  0.9166666666666666
# BaggingClassifier 의 정답률 :  1.0
# BernoulliNB 의 정답률 :  0.3888888888888889
# CalibratedClassifierCV 의 정답률 :  1.0
# CategoricalNB 의 정답률 :  0.4166666666666667
# ComplementNB 의 정답률 :  0.9166666666666666
# DecisionTreeClassifier 의 정답률 :  0.9444444444444444
# DummyClassifier 의 정답률 :  0.3888888888888889
# ExtraTreeClassifier 의 정답률 :  0.8888888888888888
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  1.0
# GaussianProcessClassifier 의 정답률 :  1.0
# GradientBoostingClassifier 의 정답률 :  0.9444444444444444
# HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
# KNeighborsClassifier 의 정답률 :  0.9444444444444444
# LabelPropagation 의 정답률 :  0.9444444444444444
# LabelSpreading 의 정답률 :  0.9444444444444444
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  1.0
# LogisticRegression 의 정답률 :  1.0
# LogisticRegressionCV 의 정답률 :  1.0
# MLPClassifier 의 정답률 :  1.0
# MultinomialNB 의 정답률 :  0.9722222222222222
# NearestCentroid 의 정답률 :  0.9722222222222222
# NuSVC 의 정답률 :  1.0
# PassiveAggressiveClassifier 의 정답률 :  1.0
# Perceptron 의 정답률 :  1.0
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
# RadiusNeighborsClassifier 의 정답률 :  0.9722222222222222
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  1.0
# RidgeClassifierCV 의 정답률 :  1.0
# SGDClassifier 의 정답률 :  1.0
# SVC 의 정답률 :  1.0