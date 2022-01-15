from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
# print(submit_file.columns)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

print(x.columns)                # column의 갯수가 8개로 줄음
print(x.shape)                  # (10886, 8)

y = train['count']
print(y)                       
print(y.shape)                  # (10886,)

print(submit_file.columns)

import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
print(y)
print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ",len(allAlgorithms))   # 모델의 갯수 :  41(classifier) / 54(regressor)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 r2값 : ', r2)
    except:
        continue
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# ARDRegression 의 r2값 :  0.27553328377320807
# AdaBoostRegressor 의 r2값 :  0.24861404169238743
# BaggingRegressor 의 r2값 :  0.2514342085964838
# BayesianRidge 의 r2값 :  0.27570882805743846
# CCA 의 r2값 :  -0.06978595926405617
# DecisionTreeRegressor 의 r2값 :  -0.12619714881396504
# DummyRegressor 의 r2값 :  -8.532574979902563e-08
# ElasticNet 의 r2값 :  0.27373957905100377
# ElasticNetCV 의 r2값 :  0.2707066666454847
# ExtraTreeRegressor 의 r2값 :  -0.09729391401661447
# ExtraTreesRegressor 의 r2값 :  0.20797222710989838
# GammaRegressor 의 r2값 :  0.2641346286185686
# GaussianProcessRegressor 의 r2값 :  -0.2860989213224898
# GradientBoostingRegressor 의 r2값 :  0.3456906997605945
# HistGradientBoostingRegressor 의 r2값 :  0.37270207646202547
# HuberRegressor 의 r2값 :  0.2467057317181407
# KNeighborsRegressor 의 r2값 :  0.2038455324514955
# KernelRidge 의 r2값 :  0.2608712275566435
# Lars 의 r2값 :  0.2757033009386364
# LarsCV 의 r2값 :  0.2751951234147687
# Lasso 의 r2값 :  0.2754765102707124
# LassoCV 의 r2값 :  0.2754232398354264
# LassoLars 의 r2값 :  -8.532574979902563e-08
# LassoLarsCV 의 r2값 :  0.2751951234147687
# LassoLarsIC 의 r2값 :  0.27567298983778266
# LinearRegression 의 r2값 :  0.2757033009386364
# LinearSVR 의 r2값 :  0.21666356032992018
# MLPRegressor 의 r2값 :  0.3064101388776792
# NuSVR 의 r2값 :  0.2216824835967366
# OrthogonalMatchingPursuit 의 r2값 :  0.16926571255409573
# OrthogonalMatchingPursuitCV 의 r2값 :  0.2740905595558095
# PLSCanonical 의 r2값 :  -0.5136954427382543
# PLSRegression 의 r2값 :  0.269035405681765
# PassiveAggressiveRegressor 의 r2값 :  0.035964051193617785
# PoissonRegressor 의 r2값 :  0.27505120976633135
# RANSACRegressor 의 r2값 :  0.16990958681838542
# RadiusNeighborsRegressor 의 r2값 :  -1.2093924824783681e+33
# RandomForestRegressor 의 r2값 :  0.2999995721317865
# Ridge 의 r2값 :  0.2757036611493413
# RidgeCV 의 r2값 :  0.2757066298387447
# SGDRegressor 의 r2값 :  -5.52988177648851
# SVR 의 r2값 :  0.20491193933055152
# TheilSenRegressor 의 r2값 :  0.265999533859674
# TransformedTargetRegressor 의 r2값 :  0.2757033009386364
# TweedieRegressor 의 r2값 :  0.27169362157941856