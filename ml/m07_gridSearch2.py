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

#2. 모델 구성
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
#                      refit=True) # refit = false 라면 AttributeError: 'GridSearchCV' object has no attribute 'best_estimator_'
model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

###########################################################
# 최적의 파라미터 확인
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

####### gridSearch 한 값! #######

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668         # train 에서의 최고값 !!!!!!!!!!여기부터 시작!!!!!!!!!!!!!
# model.score :  0.9666666666666667         # test data 에서의 최고값
# accuracy_score :  0.9666666666666667