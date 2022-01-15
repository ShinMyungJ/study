from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# ARDRegression 의 r2값 :  0.4670336652870781
# AdaBoostRegressor 의 r2값 :  0.4677727917127862
# BaggingRegressor 의 r2값 :  0.400146973812069
# BayesianRidge 의 r2값 :  0.45770277915190327
# CCA 의 r2값 :  0.4381070029563098
# DecisionTreeRegressor 의 r2값 :  0.14406021351234533
# DummyRegressor 의 r2값 :  -0.011962984778542296
# ElasticNet 의 r2값 :  0.13082214338749865
# ElasticNetCV 의 r2값 :  0.4607369144104332
# ExtraTreeRegressor 의 r2값 :  -0.42663488111753467
# ExtraTreesRegressor 의 r2값 :  0.44735380900219357
# GammaRegressor 의 r2값 :  0.08029031908102047
# GaussianProcessRegressor 의 r2값 :  -17.734682101728374
# GradientBoostingRegressor 의 r2값 :  0.44779420635961087
# HistGradientBoostingRegressor 의 r2값 :  0.37519090810376543
# HuberRegressor 의 r2값 :  0.4472595383583047
# KNeighborsRegressor 의 r2값 :  0.4503424728105596
# KernelRidge 의 r2값 :  0.45603293351121643
# Lars 의 r2값 :  -1.203367750585898
# LarsCV 의 r2값 :  0.4722505318607023
# Lasso 의 r2값 :  0.46972020530847247
# LassoCV 의 r2값 :  0.47276199464335056
# LassoLars 의 r2값 :  0.37890235603594236
# LassoLarsCV 의 r2값 :  0.4716071663733019
# LassoLarsIC 의 r2값 :  0.4712510777170408
# LinearRegression 의 r2값 :  0.45260660216173787
# LinearSVR 의 r2값 :  0.23912921470980064
# MLPRegressor 의 r2값 :  -0.41724713626695564
# NuSVR 의 r2값 :  0.15191974468247293
# OrthogonalMatchingPursuit 의 r2값 :  0.23335039815872138
# OrthogonalMatchingPursuitCV 의 r2값 :  0.46936551659312953
# PLSCanonical 의 r2값 :  -1.5249671559732398
# PLSRegression 의 r2값 :  0.4413362659940103
# PassiveAggressiveRegressor 의 r2값 :  0.4528356270073637
# PoissonRegressor 의 r2값 :  0.4441692945832215
# RANSACRegressor 의 r2값 :  0.17275364913544322
# RadiusNeighborsRegressor 의 r2값 :  0.1512475693384715
# RandomForestRegressor 의 r2값 :  0.4269088747710683
# Ridge 의 r2값 :  0.45921222867719014
# RidgeCV 의 r2값 :  0.4552971481969251
# SGDRegressor 의 r2값 :  0.4600294100926001
# SVR 의 r2값 :  0.15875529246365316
# TheilSenRegressor 의 r2값 :  0.4462514990703661
# TransformedTargetRegressor 의 r2값 :  0.45260660216173787
# TweedieRegressor 의 r2값 :  0.07699640230917337