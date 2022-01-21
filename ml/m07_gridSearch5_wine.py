# 실습

# 모델 : RandomForestClassifier

from inspect import Parameter


# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

#파라미터 조합으로 2개 이상 엮을것

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5        # 3 or 5를 많이씀
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12]},                                             # 8
    {'n_estimators' : [100,200], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},   # 32
    {'n_estimators' : [100,200], 'max_depth' : [6, 8, 10, 12],                                              # 128
     'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_leaf' : [3, 5, 7, 10]},
]                                                                                                           # 총 168

#2. 모델 구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
                     refit=True, n_jobs=-1) # refit = false 라면 AttributeError: 'GridSearchCV' object has no attribute 'best_estimator_'
# model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측

# x_test = x_train  # 과적합 상황 보여주기
# y_test = y_train  # train데이터로 best_estimator_로 예측뒤 점수를 내면
                    # best_score_ 나온다.

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)         # 이렇게하면 best_score_와 model.score 사이에서 실수할 일 없음
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", end - start)
###########################################################

# print(model.cv_results_)              # 훈련 정보가 다 나옴(42번 훈련한 것들)
aaa = pd.DataFrame(model.cv_results_)   # 깔끔하게 나옴
# print(aaa)

bbb = aaa[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score']]        # 자리가 좁아서 아래에 주석처리한것!
#         #    'split1_test_score', 'split1_test_score', 'split2_test_score',
#         #    'split3_test_score', 'split4_test_score'
#         #    ]]

print(bbb)

# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 200}
# best_score_ :  0.9857142857142858
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  10.011725425720215