from tabnanny import verbose
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv       # 정식버전이 아니라서 이 부분 포함되어야함!
from sklearn.model_selection import KFold, GridSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

n_splits = 5        # 3 or 5를 많이씀
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000, 2000], "kernel":["linear"], "degree":[3, 4, 5]},      # 15
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},               # 6
    {"C":[1, 10, 100, 200, 1000], "kernel":["sigmoid"],                         # 100
     "gamma":[0.1, 0.01, 0.02, 0.001, 0.0001], "degree":[3, 4, 5, 6]},
]                                                       # 총 42

#2. 모델 구성
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,          # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
#                      refit=True, n_jobs=-1) # refit = false 라면 AttributeError: 'GridSearchCV' object has no attribute 'best_estimator_'
# model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1,    # Random : 파라미터를 랜덤으로 뽑아서 씀
#                      refit=True, n_jobs=-1, n_iter=20, random_state=66) # n_iter의 default 값은 10
model = HalvingGridSearchCV(SVC(), parameters, cv=kfold, verbose=1,     # HalvingGridSearchCV : 데이터의 일부만 돌려서 랭크를 나누고 상위값만 뽑아서 다시 돌린다
                       refit=True, n_jobs=-1)

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
'''
# print(model.cv_results_)              # 훈련 정보가 다 나옴(42번 훈련한 것들)
aaa = pd.DataFrame(model.cv_results_)   # 깔끔하게 나옴
print(aaa)

bbb = aaa[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score']]        # 자리가 좁아서 아래에 주석처리한것!
        #    'split1_test_score', 'split1_test_score', 'split2_test_score',
        #    'split3_test_score', 'split4_test_score'
        #    ]]

print(bbb)
'''
# Gridsearchcv
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# 최적 튠 ACC :  0.9666666666666667
# 걸린시간 :  2.2789785861968994

# Randomcv
# 최적의 매개변수 :  SVC(C=200, degree=6, gamma=0.001, kernel='sigmoid')
# 최적의 파라미터 :  {'kernel': 'sigmoid', 'gamma': 0.001, 'degree': 6, 'C': 200}
# best_score_ :  0.9666666666666668
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# 최적 튠 ACC :  0.9666666666666667
# 걸린시간 :  1.9425508975982666

# 중첩교차검증
# cv를 train과 test를 1:1로 나눠서 결과를 돌리고
# 그것을 다시 cv로 돌린다