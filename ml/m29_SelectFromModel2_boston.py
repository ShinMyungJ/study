# 맹그러봐!

from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel


#1. 데이터
x, y = load_boston(return_X_y=True)

print(x.shape, y.shape)

x = x[:, [0,2,4,5,7,8,9,10,12]]          # 1, 3, 11, 6
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = PolynomialFeatures()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBRegressor(n_jobs = -1)

#3. 컴파일, 훈련

import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start


#4. 평가, 예측
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print("model.score : ", score)

# print(model.feature_importances_)
# print(np.sort(model.feature_importances_))
# thresholds = np.sort(model.feature_importances_)
# # [0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664        1, 3, 11, 6
# #  0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
# #  0.4284835 ]
# # [0.00134153 0.00363372 0.01203114 0.01220458 0.01447933 0.01479118
# #  0.0175432  0.03041654 0.04246344 0.05182539 0.06949984 0.30128664
# #  0.4284835 ]

# print("=============================================")
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh,        # feature_importances_ 수치가 낮은 칼럼부터 빼기 시작
#                                 prefit=True)                    # 수치 낮은 칼럼 하나씩 더 빼서 r2 score를 구함
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
    
#     print("Thresh=%.3f, n=%d, R2: %.2f%%"
#           %(thresh, select_x_train.shape[1], score*100))


# 피처 줄인다음에!!!
# 다시 모델해서 결과 비교
# (404, 13) (102, 13)
# Thresh=0.001, n=13, R2: 92.21%
# (404, 12) (102, 12)
# Thresh=0.004, n=12, R2: 92.16%
# (404, 11) (102, 11)
# Thresh=0.012, n=11, R2: 92.03%
# (404, 10) (102, 10)
# Thresh=0.012, n=10, R2: 92.19%
# (404, 9) (102, 9)
# Thresh=0.014, n=9, R2: 93.08%
# (404, 8) (102, 8)
# Thresh=0.015, n=8, R2: 92.37%
# (404, 7) (102, 7)
# Thresh=0.018, n=7, R2: 91.48%
# (404, 6) (102, 6)
# Thresh=0.030, n=6, R2: 92.71%
# (404, 5) (102, 5)
# Thresh=0.042, n=5, R2: 91.74%
# (404, 4) (102, 4)
# Thresh=0.052, n=4, R2: 92.11%
# (404, 3) (102, 3)
# Thresh=0.069, n=3, R2: 92.52%
# (404, 2) (102, 2)
# Thresh=0.301, n=2, R2: 69.41%
# (404, 1) (102, 1)
# Thresh=0.428, n=1, R2: 44.98%

# model.score :  0.9221188601856797

# 피쳐 줄인 후
# model.score :  0.9307724288278274