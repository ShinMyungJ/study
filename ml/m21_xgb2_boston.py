# 기존 모델결과와 비교

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
                     n_estimators = 3500,
                     learning_rate = 0.01,
                     max_depth = 3,
                     min_child_weight = 1,
                     subsample = 1,
                     colsample_bytree = 1,
                     reg_alpha = 0,         # 규제  L1
                     reg_lambda = 1,        # 규제  L2
                     verbose=1)

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start


#4. 평가, 예측

result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("XGBRegressor : ", round(result, 4))
print("걸린시간 : ", round(end, 4),"초")
print("r2 : ", round(r2, 4))
# print(model.feature_importances_)


# XGBRegressor :  0.9434
# 걸린시간 :  1.6901 초
# r2 :  0.9434

# DecisionTreeRegressor :  0.9333333333333333
# r2uracy_score :  0.9333333333333333
# [0.0125026  0.         0.53835801 0.44913938]

# RandomForestRegressor :  0.9
# r2uracy_score :  0.9
# [0.10013035 0.02902692 0.42459887 0.44624385]

# GradientBoostingRegressor :  0.9666666666666667
# r2uracy_score :  0.9666666666666667
# [0.00784771 0.01012783 0.24292126 0.73910321]

# XGBRegressor :  0.9
# r2uracy_score :  0.9
# [0.01835513 0.0256969  0.6204526  0.33549538]


# 결과 비교
# 1. DecisionTree
# 기존 r2 : 0.9333333333333333
# 컬럼삭제 후 r2 : 0.9333333333333333

# 2. RandomForestClassifier
# 기존 r2 : 0.9
# 컬럼삭제 후 r2 : 0.9666666666666667

# 3. GradientBoostingClassifier
# 기존 r2 : 0.9666666666666667
# 컬럼삭제 후 r2 : 0.9333333333333333

# 4. XGBClassifier
# 기존 r2 : 0.9
# 컬럼삭제 후 r2 : 0.9666666666666667

# QuantileTransformer
# XGBRegressor :  0.9352222946636575

# PowerTransformer
# XGBRegressor :  0.9313452507052867

# PolynomialFeatures
# XGBRegressor :  0.9099174326995426