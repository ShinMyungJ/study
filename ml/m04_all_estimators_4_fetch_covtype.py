from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = fetch_covtype()

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
        # continue
        print(name, '은 에러 터진 놈!!!!')
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# AdaBoostClassifier 의 정답률 :  0.4927755737803671
# BaggingClassifier 의 정답률 :  0.9601817509014398
# BernoulliNB 의 정답률 :  0.6300956085471116
# CalibratedClassifierCV 의 정답률 :  0.7128043165839092
# CategoricalNB 의 정답률 :  0.6304828618882473
# ComplementNB 의 정답률 :  0.6184694026832354
# DecisionTreeClassifier 의 정답률 :  0.9389688734370025
# DummyClassifier 의 정답률 :  0.48621808387046805
# ExtraTreeClassifier 의 정답률 :  0.8671290758414154
# ExtraTreesClassifier 의 정답률 :  0.9525055291171485
# GaussianNB 의 정답률 :  0.08926619794669673
# HistGradientBoostingClassifier 의 정답률 :  0.7828369319208627
# KNeighborsClassifier 의 정답률 :  0.9358364241887043
# LinearDiscriminantAnalysis 의 정답률 :  0.6782785298142044
# LinearSVC 의 정답률 :  0.7126752321368639
# LogisticRegression 의 정답률 :  0.7201879469548979
# LogisticRegressionCV 의 정답률 :  0.7241637479238918
# MLPClassifier 의 정답률 :  0.8385153567463834
# MultinomialNB 의 정답률 :  0.6405600543875803
# NearestCentroid 의 정답률 :  0.3849814548677745
# PassiveAggressiveClassifier 의 정답률 :  0.6681410979062503
# Perceptron 의 정답률 :  0.4591619837697822
# QuadraticDiscriminantAnalysis 의 정답률 :  0.0829754825606912
# RidgeClassifier 의 정답률 :  0.7012125332392452
# RidgeClassifierCV 의 정답률 :  0.7012469557584572
# SGDClassifier 의 정답률 :  0.7128559503627273