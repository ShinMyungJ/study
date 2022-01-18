import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, Perceptron
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
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

y = train['count']
print(y)                       
print(y.shape)

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

# 모델의 갯수 :  54
# ARDRegression 의 정답률 :  [0.26718059 0.236777   0.24216522 0.25623614 0.27155516]
# AdaBoostRegressor 의 정답률 :  [0.20980263 0.21065152 0.21120638 0.19821059 0.1910902 ]
# BaggingRegressor 의 정답률 :  [0.21800782 0.23763778 0.22765912 0.26810834 0.2085698 ]
# BayesianRidge 의 정답률 :  [0.2663943  0.23615078 0.24239991 0.25680871 0.27222936]
# CCA 의 정답률 :  [-0.11593223 -0.17613333 -0.11929902 -0.10816997 -0.11657828]
# DecisionTreeRegressor 의 정답률 :  [-0.23043932 -0.13483162 -0.18766561 -0.21089725 -0.27682711]
# DummyRegressor 의 정답률 :  [-0.00036479 -0.00352618 -0.00257159 -0.00135553 -0.00337107]
# ElasticNet 의 정답률 :  [0.05860751 0.05221671 0.0540052  0.05761403 0.05743841]
# ElasticNetCV 의 정답률 :  [0.25302879 0.22906229 0.23453173 0.24579338 0.25786245]
# ExtraTreeRegressor 의 정답률 :  [-0.17403466 -0.11487841 -0.16502144 -0.12329055 -0.2159581 ]
# ExtraTreesRegressor 의 정답률 :  [0.17581664 0.21666298 0.20305564 0.20599358 0.13551406]
# GammaRegressor 의 정답률 :  [0.0219256  0.01925851 0.02111146 0.02120952 0.0204103 ]
# GaussianProcessRegressor 의 정답률 :  [ -5.64258163  -5.48758184  -5.15604839 -44.303364    -5.52046105]
# GradientBoostingRegressor 의 정답률 :  [0.33025469 0.3104335  0.30795495 0.3289654  0.3437582 ]
# HistGradientBoostingRegressor 의 정답률 :  [0.35218761 0.33932504 0.326562   0.35363908 0.35432011]
# HuberRegressor 의 정답률 :  [0.24447819 0.1994666  0.2036237  0.24050791 0.26473932]
# IsotonicRegression 은 에러 터진 놈!!!!
# KNeighborsRegressor 의 정답률 :  [0.27133485 0.29642002 0.2534889  0.3022791  0.27597115]
# KernelRidge 의 정답률 :  [0.24017669 0.21264308 0.21321056 0.22275262 0.24798623]
# Lars 의 정답률 :  [0.26660729 0.23619668 0.24226355 0.25656582 0.27226183]
# LarsCV 의 정답률 :  [0.2665705  0.23619668 0.24272762 0.25658719 0.27053345]
# Lasso 의 정답률 :  [0.26149731 0.23570037 0.24229754 0.25459082 0.26619054]
# LassoCV 의 정답률 :  [0.26651536 0.23623885 0.24273815 0.25667688 0.27143031]
# LassoLars 의 정답률 :  [-0.00036479 -0.00352618 -0.00257159 -0.00135553 -0.00337107]
# LassoLarsCV 의 정답률 :  [0.2665705  0.23619668 0.24272762 0.25658719 0.27053345]
# LassoLarsIC 의 정답률 :  [0.26660729 0.2370822  0.24260298 0.25662591 0.26973167]
# LinearRegression 의 정답률 :  [0.26660729 0.23619668 0.24226355 0.25656582 0.27226183]
# LinearSVR 의 정답률 :  [0.18707012 0.14615314 0.14769718 0.18723988 0.21495547]
# MLPRegressor 의 정답률 :  [0.26669549 0.23702969 0.2417848  0.25420076 0.27269643]
# MultiOutputRegressor 은 에러 터진 놈!!!!
# MultiTaskElasticNet 은 에러 터진 놈!!!!
# MultiTaskElasticNetCV 은 에러 터진 놈!!!!
# MultiTaskLasso 은 에러 터진 놈!!!!
# MultiTaskLassoCV 은 에러 터진 놈!!!!
# NuSVR 의 정답률 :  [0.19681332 0.16535507 0.16651475 0.20169695 0.2146087 ]
# OrthogonalMatchingPursuit 의 정답률 :  [0.15489456 0.14813388 0.15182398 0.14409813 0.15201417]
# OrthogonalMatchingPursuitCV 의 정답률 :  [0.26252578 0.23347235 0.2410801  0.2564686  0.26854405]
# PLSCanonical 의 정답률 :  [-0.52365058 -0.58845465 -0.57020911 -0.65055547 -0.61930074]
# PLSRegression 의 정답률 :  [0.26105692 0.23150838 0.2350149  0.25099901 0.26358952]
# PassiveAggressiveRegressor 의 정답률 :  [0.24770024 0.20415024 0.1696432  0.18870133 0.21927658]
# PoissonRegressor 의 정답률 :  [0.27202789 0.24117019 0.25133601 0.26420168 0.2770226 ]
# RANSACRegressor 의 정답률 :  [ 0.1929769   0.00504051  0.0267548  -0.08333493 -0.15123151]
# RadiusNeighborsRegressor 의 정답률 :  [0.06235578 0.05739043 0.05930647 0.05814315 0.05683601]
# RandomForestRegressor 의 정답률 :  [0.27558928 0.286701   0.27619345 0.31405986 0.25900364]
# RegressorChain 은 에러 터진 놈!!!!
# Ridge 의 정답률 :  [0.26642969 0.23615529 0.24238848 0.25678794 0.27223924]
# RidgeCV 의 정답률 :  [0.26642969 0.23615529 0.24238848 0.25678794 0.27223924]
# SGDRegressor 의 정답률 :  [0.26567048 0.23492153 0.24152032 0.25643485 0.27159062]
# SVR 의 정답률 :  [0.19240348 0.15140308 0.15125877 0.19735371 0.21975808]
# StackingRegressor 은 에러 터진 놈!!!!
# TheilSenRegressor 의 정답률 :  [0.25392972 0.21771557 0.22521215 0.2506491  0.27232683]
# TransformedTargetRegressor 의 정답률 :  [0.26660729 0.23619668 0.24226355 0.25656582 0.27226183]
# TweedieRegressor 의 정답률 :  [0.03408486 0.02915555 0.03057026 0.03319779 0.03217802]
# VotingRegressor 은 에러 터진 놈!!!!