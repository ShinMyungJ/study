# 기존 모델결과와 비교

from lightgbm import early_stopping
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

# datasets = fetch_california_housing()
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']

import pickle
path = './_save/'
pickle.dump(datasets, open(path +
                           'm26_pickle1_save_dataset.dat', 'wb'))

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
model = XGBClassifier(n_jobs=-1,
                     n_estimators = 1000,
                     learning_rate = 0.025,
                     max_depth = 3,
                     min_child_weight = 1,
                     subsample = 1,
                     colsample_bytree = 1,
                     reg_alpha = 0,         # 규제  L1
                     reg_lambda = 0,        # 규제  L2
                     )

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='mlogloss',              # rmse, mae, logloss, error
          early_stopping_rounds=10,        # mlogloss, merror 
          )     

end = time.time() - start


#4. 평가, 예측

results = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("XGBRegressor : ", round(results, 4))
print("걸린시간 : ", round(end, 4),"초")
print("accuracy_score : ", round(acc, 4))

# print("===========================================")
# hist = model.evals_result()
# print(hist)

#저장
# import pickle
# path = './_save/'
# pickle.dump(model, open(path + 'm26_pickle1_save', 'wb'))
