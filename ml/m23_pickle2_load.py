# 기존 모델결과와 비교

from lightgbm import early_stopping
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

# datasets = fetch_california_housing()
datasets = load_boston()

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

#불러오기 //2. 모델, 3. 훈련
import pickle
path = './_save/'
model = pickle.load(open(path + 'm23_pickle1_save.dat', 'rb'))

#4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print(r2)

print("===========================================")
hist = model.evals_result()
print(hist)

# print(model.feature_importances_)

# import pickle
# path = './_save/'
# pickle.dump(model, open(path + 'm23_pickle1_save', 'wb'))
