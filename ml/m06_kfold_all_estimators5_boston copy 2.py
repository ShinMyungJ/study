from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_boston()

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
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

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

# 모델의 갯수 :  55
# ARDRegression 의 정답률 :  [0.72766314 0.70095126 0.82947432 0.71316178 0.60634586]
# AdaBoostRegressor 의 정답률 :  [0.78184355 0.74124669 0.84731498 0.81969165 0.74752391]
# BaggingRegressor 의 정답률 :  [0.81335265 0.8044926  0.89793698 0.79971669 0.72385751]
# BayesianRidge 의 정답률 :  [0.73036074 0.70326995 0.81931675 0.7276768  0.62243287]
# CCA 의 정답률 :  [0.71381738 0.57527041 0.76550464 0.627568   0.50654813]
# DecisionTreeRegressor 의 정답률 :  [0.69019749 0.32213121 0.85755014 0.45408871 0.68084402]
# DummyRegressor 의 정답률 :  [-0.01786394 -0.05727721 -0.00078388 -0.01713141 -0.00998821]
# ElasticNet 의 정답률 :  [0.11072655 0.16354151 0.18483752 0.1898607  0.09204848]
# ElasticNetCV 의 정답률 :  [0.72590969 0.70402096 0.82199949 0.73227373 0.62313516]
# ExtraTreeRegressor 의 정답률 :  [0.4983724  0.48391386 0.70484575 0.589567   0.69178718]
# ExtraTreesRegressor 의 정답률 :  [0.85033967 0.80012242 0.91169605 0.88365532 0.83644034]
# GammaRegressor 의 정답률 :  [0.16726264 0.18612106 0.20522873 0.22509375 0.12464323]
# GaussianProcessRegressor 의 정답률 :  [ 0.02056552 -1.30414363  0.16945239 -3.24990656  0.16050144]
# GradientBoostingRegressor 의 정답률 :  [0.87374475 0.79353888 0.91984559 0.8752238  0.80951757]
# HistGradientBoostingRegressor 의 정답률 :  [0.81108915 0.80579998 0.89467439 0.82665421 0.85340541]
# HuberRegressor 의 정답률 :  [0.69286991 0.74327339 0.81340068 0.79361147 0.56976525]
# IsotonicRegression 은 에러 터진 놈!!!!
# KNeighborsRegressor 의 정답률 :  [0.70387438 0.33564282 0.72233092 0.7931044  0.62932915]
# KernelRidge 의 정답률 :  [0.67823496 0.55036871 0.76738051 0.7367567  0.53554151]
# Lars 의 정답률 :  [0.73163895 0.70204718 0.81549312 0.7226594  0.62028649]
# LarsCV 의 정답률 :  [0.73040458 0.70204718 0.82120263 0.72500121 0.62028649]
# Lasso 의 정답률 :  [0.17704832 0.23215058 0.28854968 0.27967835 0.13400164]
# LassoCV 의 정답률 :  [0.73075575 0.7026334  0.81668778 0.72333703 0.61939287]
# LassoLars 의 정답률 :  [-0.01786394 -0.05727721 -0.00078388 -0.01713141 -0.00998821]
# LassoLarsCV 의 정답률 :  [0.73131115 0.70204718 0.81549312 0.7226594  0.62028649]
# LassoLarsIC 의 정답률 :  [0.73115855 0.70408636 0.82130896 0.72661725 0.61483267]
# LinearRegression 의 정답률 :  [0.73163895 0.70204718 0.81549312 0.7226594  0.62028649]
# LinearSVR 의 정답률 :  [0.57231187 0.66210275 0.67237409 0.74054521 0.47297922]
# MLPRegressor 의 정답률 :  [0.22125405 0.0439991  0.08823515 0.3098099  0.08198045]
# MultiOutputRegressor 은 에러 터진 놈!!!!
# MultiTaskElasticNet 은 에러 터진 놈!!!!
# MultiTaskElasticNetCV 은 에러 터진 놈!!!!
# MultiTaskLasso 은 에러 터진 놈!!!!
# MultiTaskLassoCV 은 에러 터진 놈!!!!
# NuSVR 의 정답률 :  [0.46755909 0.59387976 0.63737894 0.6861065  0.44572846]
# OrthogonalMatchingPursuit 의 정답률 :  [0.50996533 0.38217578 0.64132423 0.49642429 0.37274419]
# OrthogonalMatchingPursuitCV 의 정답률 :  [0.71404495 0.64343077 0.81974013 0.70052564 0.55958834]
# PLSCanonical 의 정답률 :  [-1.42085198 -4.47397236 -2.06827749 -2.50983396 -1.27278486]
# PLSRegression 의 정답률 :  [0.71236784 0.68302237 0.82707472 0.76221139 0.57803964]
# PassiveAggressiveRegressor 의 정답률 :  [0.69804682 0.70775883 0.73662375 0.6142928  0.48419903]
# PoissonRegressor 의 정답률 :  [0.62397616 0.64337594 0.70161094 0.72713591 0.510408  ]
# QuantileRegressor 의 정답률 :  [-0.05311999 -0.00125696 -0.03306242 -0.00082186 -0.03432251]
# RANSACRegressor 의 정답률 :  [ 0.70088337 -0.38778964  0.7304011   0.6966428   0.39966881]
# RadiusNeighborsRegressor 의 정답률 :  [ 0.39236877 -0.16285928  0.43289939  0.45328512  0.24597178]
# RandomForestRegressor 의 정답률 :  [0.84151495 0.82003684 0.91109956 0.83011012 0.75009402]
# RegressorChain 은 에러 터진 놈!!!!
# Ridge 의 정답률 :  [0.72208679 0.7038357  0.82210249 0.73896938 0.62255537]
# RidgeCV 의 정답률 :  [0.73107343 0.70268707 0.81746509 0.72523793 0.62174712]
# SGDRegressor 의 정답률 :  [0.72292194 0.64729445 0.82061017 0.75158986 0.60678056]
# SVR 의 정답률 :  [0.49215097 0.62787465 0.66338005 0.70918557 0.47475133]
# StackingRegressor 은 에러 터진 놈!!!!
# TheilSenRegressor 의 정답률 :  [0.68681268 0.76185747 0.8045377  0.78051151 0.5652636 ]
# TransformedTargetRegressor 의 정답률 :  [0.73163895 0.70204718 0.81549312 0.7226594  0.62028649]
# TweedieRegressor 의 정답률 :  [0.14133742 0.19066949 0.20746253 0.22331937 0.10919935]
# VotingRegressor 은 에러 터진 놈!!!!