# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time
# import warnings
# warnings.filterwarnings('ignore')

#1. 데이터

datasets = fetch_california_housing()
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
model = XGBRegressor(
                     n_jobs=-1,
                    #  verbose=1,
                     n_estimators = 100,
                     learning_rate = 0.04,
                     max_depth = 6,
                     min_child_weight = 1,
                     subsample = 1,
                     colsample_bytree = 0.6,
                     reg_alpha = 1,         # 규제  L1
                     reg_lambda = 1,        # 규제  L2
                     )

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse',              # rmse, mae, logloss, error
          )     # mlogloss, merror
end = time.time() - start


#4. 평가, 예측

results = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("XGBRegressor : ", round(results, 4))
print("걸린시간 : ", round(end, 4),"초")
print("r2 : ", round(r2, 4))
# print(model.feature_importances_)


# XGBRegressor :  0.8625
# 걸린시간 :  23.2732 초
# r2 :  0.8625

# default
# XGBRegressor :  0.8434
# 걸린시간 :  1.0253 초
# r2 :  0.8434

print("===========================================")
hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt

train_error = hist['validation_0']['rmse']
test_error = hist['validation_1']['rmse']
print(train_error)

loss1 = hist.get('validation_0').get('rmse')
loss2 = hist.get('validation_1').get('rmse')
plt.plot(loss1, 'y--', label="training loss")
plt.plot(loss2, 'r--', label="test loss")
plt.grid()
plt.legend()
plt.show()

# epoch = range(1, len(train_error)+1)
# plt.plot(epoch, train_error, label = 'Train')
# plt.plot(epoch, test_error, label = 'Test')
# plt.ylabel('Classification Error')
# plt.xlabel('Model Complexity (n_estimators)')
# plt.legend()
# plt.show()

# plt.figure(figsize=(9,6))
# plt.plot(hist['validation_0']['mae'], marker=".", c='red', label='train_set')
# plt.plot(hist['validation_1']['mae'], marker='.', c='blue', label='test_set')
# plt.grid() 
# plt.title('loss_mae')
# plt.ylabel('loss_mae')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') 
# plt.show()

# print("=================================================")
# print(hist)
# print("=================================================")
# print(hist.history)
# print("=================================================")
# print(hist.history['loss'])
# print("=================================================")
# print(hist.history['val_loss'])
# print("=================================================")

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,5))

# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')          # 선을 긋는다
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()                  # 점자 형태
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.show()