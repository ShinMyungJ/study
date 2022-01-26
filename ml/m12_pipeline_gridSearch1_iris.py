import numpy as np
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# parameters = [
#     {'randomforestclassifier__max_depth' : [6, 8, 10],
#      'randomforestclassifier__min_samples_leaf' : [3, 5, 7]},
#     {'randomforestclassifier__min_samples_leaf' : [3, 5, 7],
#      'randomforestclassifier__min_samples_split' : [3, 5, 10]}
# ]

parameters = [
    {'rf__max_depth' : [6, 8, 10],
     'rf__min_samples_leaf' : [3, 5, 7]},
    {'rf__min_samples_leaf' : [3, 5, 7],
     'rf__min_samples_split' : [3, 5, 10]}
]

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA       # PCA : 고차원의 데이터를 저차원의 데이터로 환원, 선을 하나 그어서 거기에 맞춤, 컬럼을 압축(차원 축소)
                                            # fit만 가능

# model = SVC()
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([("mm", MinMaxScaler()),("rf",RandomForestClassifier())])

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start


#4. 평가, 예측
result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("걸린시간 : ", round(end, 3), '초')
print("model.score : ", result)
print("accuracy_score : ", acc)

# ===================================RandomForestClassifier===================================

# GridSearchCV
# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# 걸린시간 :  7.935 초
# model.score :  1.0
# accuracy_score :  1.0

# RandomizedSearchCV
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 걸린시간 :  4.514 초
# model.score :  1.0
# accuracy_score :  1.0

# HalvingGridSearchCV
# iter: 1
# n_candidates: 6
# n_resources: 90
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 걸린시간 :  11.127 초
# model.score :  1.0
# accuracy_score :  1.0


