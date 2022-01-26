import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

datasets = load_diabetes()
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

# model.score :  0.40315229129742514

# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=3)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 3, 'n_estimators': 100}
# best_score_ :  0.5018391479878967
# model.score :  0.3869100392330088
# accuracy_score :  0.3869100392330088
# 최적 튠 ACC :  0.3869100392330088
# 걸린시간 :  11.17442011833191

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 3}
# best_score_ :  0.49865728464672066
# model.score :  0.3696157036388863
# r2_score :  0.3696157036388863
# 최적 튠 r2 :  0.3696157036388863
# 걸린시간 :  3.7316458225250244

# HalvingGridSearchCV
# iter: 3
# n_candidates: 3
# n_resources: 351
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score_ :  0.5001539990483752
# model.score :  0.38532743469820174
# r2_score :  0.38532743469820174
# 최적 튠 r2 :  0.38532743469820174
# 걸린시간 :  14.146950244903564