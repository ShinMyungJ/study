# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_breast_cancer()
# datasets = load_wine()
# datasets = fetch_california_housing()
# datasets = load_boston()

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = RobustScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
                     n_jobs=-1,
                     n_estimators = 3000,
                     learning_rate = 0.01,
                     max_depth = 4,
                     min_child_weight = 1,
                     subsample = 1,
                     colsample_bytree = 0.6,
                     reg_alpha = 1,         # 규제  L1
                     reg_lambda = 1,        # 규제  L2
                     )

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='error')         # mlogloss
end = time.time() - start


#4. 평가, 예측

result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("XGBClassifier : ", round(result, 4))
print("걸린시간 : ", round(end, 4),"초")
print("accuracy_score : ", round(acc, 4))

# XGBRegressor :  0.8625
# 걸린시간 :  23.2732 초
# r2 :  0.8625

# default
# XGBRegressor :  0.8434
# 걸린시간 :  1.0253 초
# r2 :  0.8434
