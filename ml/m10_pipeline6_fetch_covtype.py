import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

datasets = fetch_covtype()
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

# model.score :  0.9551216405772657

# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200}
# best_score_ :  0.9392244968309751
# model.score :  0.945784532240992
# accuracy_score :  0.945784532240992
# 최적 튠 ACC :  0.945784532240992
# 걸린시간 :  6070.5769329071045

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 5}
# best_score_ :  0.9288438883961341
# model.score :  0.935733156631068
# accuracy_score :  0.935733156631068
# 최적 튠 ACC :  0.935733156631068
# 걸린시간 :  1422.0262701511383

# HalvingGridSearchCV
# iter: 3
# n_candidates: 3
# n_resources: 464805
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5,
#                        n_estimators=200)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 200}
# best_score_ :  0.9391679052952101
# model.score :  0.9443473920638882
# accuracy_score :  0.9443473920638882
# 최적 튠 ACC :  0.9443473920638882
# 걸린시간 :  1517.9810707569122