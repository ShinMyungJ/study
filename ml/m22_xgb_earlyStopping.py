# 기존 모델결과와 비교

from lightgbm import early_stopping
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time
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

#2. 모델구성
model = XGBRegressor(n_jobs=-1,
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
          eval_metric='rmse',              # rmse, mae, logloss, error
          early_stopping_rounds=10,        # mlogloss, merror 
          )     

end = time.time() - start


#4. 평가, 예측

results = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("XGBRegressor : ", round(results, 4))
print("걸린시간 : ", round(end, 4),"초")
print("r2 : ", round(r2, 4))

print("===========================================")
hist = model.evals_result()
print(hist)
# print(model.feature_importances_)

import matplotlib.pyplot as plt

train_error = hist['validation_0']['rmse']
test_error = hist['validation_1']['rmse']

epoch = range(1, len(train_error)+1)
plt.plot(epoch, train_error, label = 'Train')
plt.plot(epoch, test_error, label = 'Test')
plt.ylabel('Classification Error')
plt.xlabel('Model Complexity (n_estimators)')
plt.legend()
plt.show()

# plt.figure(figsize=(9,6))
# plt.plot(hist['validation_0']['mae'], marker=".", c='red', label='train_set')
# plt.plot(hist['validation_1']['mae'], marker='.', c='blue', label='test_set')
# plt.grid() 
# plt.title('loss_mae')
# plt.ylabel('loss_mae')
# plt.xlabel('epoch')
# plt.legend(loc='upper right') 
# plt.show()


# XGBRegressor :  0.9434
# 걸린시간 :  1.6901 초
# r2 :  0.9434

# DecisionTreeRegressor :  0.9333333333333333
# r2_score :  0.9333333333333333
# [0.0125026  0.         0.53835801 0.44913938]

# RandomForestRegressor :  0.9
# r2_score :  0.9
# [0.10013035 0.02902692 0.42459887 0.44624385]

# GradientBoostingRegressor :  0.9666666666666667
# r2_score :  0.9666666666666667
# [0.00784771 0.01012783 0.24292126 0.73910321]

# XGBRegressor :  0.9
# r2_score :  0.9
# [0.01835513 0.0256969  0.6204526  0.33549538]

