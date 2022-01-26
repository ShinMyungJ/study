import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']

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

# model.score :  0.2948220870432797

# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score_ :  0.3527651940293491
# model.score :  0.3708706087344301
# r2_score :  0.3708706087344301
# 최적 튠 r2 :  0.3708706087344301
# 걸린시간 :  53.794854402542114

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=5, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_leaf': 5, 'max_depth': 10}
# best_score_ :  0.3527854084332157
# model.score :  0.36886935761522566
# r2_score :  0.36886935761522566
# 최적 튠 r2 :  0.36886935761522566
# 걸린시간 :  10.23756718635559

# HalvingGridSearchCV
# iter: 3
# n_candidates: 3
# n_resources: 8694
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=7, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 7, 'n_estimators': 200}
# best_score_ :  0.3512472938144575
# model.score :  0.3665435911992384
# r2_score :  0.3665435911992384
# 최적 튠 r2 :  0.3665435911992384
# 걸린시간 :  23.21535325050354