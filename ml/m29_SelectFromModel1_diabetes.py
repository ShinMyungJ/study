from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel


#1. 데이터
x, y = load_diabetes(return_X_y=True)
x = np.delete(x, [0, 1, 4, 7], axis=1)

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

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
# [0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739        # 0, 1, 7, 4
#  0.07382318 0.03284872 0.39979857 0.06597802]
thresholds = np.sort(model.feature_importances_)
# [0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
#  0.06597802 0.07382318 0.19681741 0.39979857]
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


# print("r2 score : ", r2_score(y_test, y_predict))

# XGBClassifier :  0.7286
# 걸린시간 :  16.8657 초
# accuracy_score :  0.7286

# (353, 10) (89, 10)
# Thresh=0.026, n=10, R2: 23.80%
# (353, 9) (89, 9)
# Thresh=0.033, n=9, R2: 27.04%
# (353, 8) (89, 8)
# Thresh=0.038, n=8, R2: 24.23%
# (353, 7) (89, 7)
# Thresh=0.048, n=7, R2: 26.73%
# (353, 6) (89, 6)
# Thresh=0.055, n=6, R2: 30.09%
# (353, 5) (89, 5)
# Thresh=0.063, n=5, R2: 27.58%
# (353, 4) (89, 4)
# Thresh=0.066, n=4, R2: 29.68%
# (353, 3) (89, 3)
# Thresh=0.074, n=3, R2: 23.51%
# (353, 2) (89, 2)
# Thresh=0.197, n=2, R2: 12.78%
# (353, 1) (89, 1)
# Thresh=0.400, n=1, R2: 2.56%

# 칼럼 뺀 후
# model.score :  0.3008522300150823

# 0.51