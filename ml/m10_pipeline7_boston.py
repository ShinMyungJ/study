import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

datasets = load_boston()
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
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("model.score : ", result)

# model.score :  0.8829847811408518

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668         # train 에서의 최고값 !!!!!!!!!!여기부터 시작!!!!!!!!!!!!!
# model.score :  0.9666666666666667         # test data 에서의 최고값
# accuracy_score :  0.9666666666666667

# RandomSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 12}
# best_score_ :  0.8072409831192224
# model.score :  0.9190206801322907
# R2 :  0.9190206801322907
# 최적 튠 R2 :  0.9190206801322907
# 걸린시간 :  4.2517290115356445

# HalvingGridSearchCV
# iter: 3
# n_candidates: 3
# n_resources: 378
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 8, 'n_estimators': 200}
# best_score_ :  0.8161469588637352
# model.score :  0.923687251536325
# R2 :  0.923687251536325
# 최적 튠 R2 :  0.923687251536325
# 걸린시간 :  13.8076171875