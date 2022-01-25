import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

# model = SVC()
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ", result)

# model.score :  0.9649122807017544

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668         # train 에서의 최고값 !!!!!!!!!!여기부터 시작!!!!!!!!!!!!!
# model.score :  0.9666666666666667         # test data 에서의 최고값
# accuracy_score :  0.9666666666666667

# RandomSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 8}
# best_score_ :  0.9626373626373625
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적 튠 ACC :  0.9649122807017544
# 걸린시간 :  3.9435501098632812

# HalvingGrid
# ----------
# iter: 2
# n_candidates: 8
# n_resources: 180
# Fitting 5 folds for each of 8 candidates, totalling 40 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=10,
#                        n_estimators=200)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 200}
# best_score_ :  0.9555555555555555
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적 튠 ACC :  0.9649122807017544
# 걸린시간 :  15.196932315826416
