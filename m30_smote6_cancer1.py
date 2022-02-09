# smote 넣어서 맹그러
# 넣은거 안넣은거 비교!!

from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score
from pprint import pprint
from imblearn.over_sampling import SMOTE

import pandas as pd

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(pd.Series(y).value_counts())

'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# for index, value in enumerate(y):
#     if value == 9 :
#         y[index] = 8
#     elif value == 8 :
#         y[index] = 8
#     elif value == 7 :
#         y[index] = 7
#     elif value == 6 :
#         y[index] = 6
#     elif value == 5 :
#         y[index] = 5
#     elif value == 4 :
#         y[index] = 4
#     elif value == 3 :
#         y[index] = 3
#     else:
#         y[index] = 0
        
print(pd.Series(y).value_counts())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=66)

print(pd.Series(y_train).value_counts())
smote = SMOTE(random_state=66, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)

#2. 모델
model = XGBClassifier(
                    #  n_jobs=-1,
                     n_estimators = 5000,
                     learning_rate = 0.1,
                     max_depth = 6,
                     min_child_weight = 1,
                     subsample = 0.5,
                     colsample_bytree = 1,
                     reg_alpha = 1,         # 규제  L1
                     reg_lambda = 0,        # 규제  L2
                     tree_method = 'gpu_hist',
                     predictor = 'gpu_predictor',
                     gpu_id=0,
                      )

#3. 훈련
model.fit(x_train, y_train, eval_metric='merror')

#4. 평가, 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('라벨 : ', np.unique(y, return_counts=True))
# 라벨 :  (array([ 3.,  4.,   5.,   6.,   7.,   8.,   9.]),
#          array([20,  163, 1457, 2198,  880,  175,   5], dtype=int64))
print("model.score : ", score)
print("accuracy score : ", round(accuracy_score(y_test, y_predict),4))
print("f1 score : ", round(f1_score(y_test, y_predict, average='macro'),4))
'''