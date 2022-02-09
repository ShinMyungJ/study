# 그냥 증폭해서 성능비교

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score
from pprint import pprint
from imblearn.over_sampling import SMOTE

import pandas as pd

#1. 데이터 
path = '../_data/winequlity/'   
datasets = pd.read_csv(path+'winequality-white.csv',
                       index_col=None, sep=';', header=0, dtype=float)

datasets = datasets.values      # numpy로 변환
# print(type(datasets))
# print(datasets.shape)

x = datasets[:, :-1]
y = datasets[:, -1]
# print(x.shape, y.shape)
# y = np.where(y <= 5, 5, y)
# y = np.where(y >= 7, 7, y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, stratify=y, shuffle=True, random_state=66)

print(pd.Series(y_train).value_counts())
smote = SMOTE(random_state=66, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())


# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(x_train.shape, x_test.shape)

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
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
               
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | : or , 조건에 맞는 값의 위치값

outlier_loc = outliers(datasets)
print("이상치의 위치 : ", outlier_loc)

# 시각화

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.boxplot(datasets, sym='bo')
# # plt.boxplot(aaa, notch=1, sym='b*', vert=0)
# # sns.boxplot(data = aaa)
# plt.title('Box plot of aaa')
# plt.show()

#                       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide   density        pH  sulphates   alcohol   quality
# fixed acidity              1.000000         -0.022697     0.289181        0.089021   0.023086            -0.049396              0.091070  0.265331 -0.425858  -0.017143 -0.120881 -0.113663
# volatile acidity          -0.022697          1.000000    -0.149472        0.064286   0.070512            -0.097012              0.089261  0.027114 -0.031915  -0.035728  0.067718 -0.194723
# citric acid                0.289181         -0.149472     1.000000        0.094212   0.114364             0.094077              0.121131  0.149503 -0.163748   0.062331 -0.075729 -0.009209
# residual sugar             0.089021          0.064286     0.094212        1.000000   0.088685             0.299098              0.401439  0.838966 -0.194133  -0.026664 -0.450631 -0.097577
# chlorides                  0.023086          0.070512     0.114364        0.088685   1.000000             0.101392              0.198910  0.257211 -0.090439   0.016763 -0.360189 -0.209934
# free sulfur dioxide       -0.049396         -0.097012     0.094077        0.299098   0.101392             1.000000              0.615501  0.294210 -0.000618   0.059217 -0.250104  0.008158
# total sulfur dioxide       0.091070          0.089261     0.121131        0.401439   0.198910             0.615501              1.000000  0.529881  0.002321   0.134562 -0.448892 -0.174737
# density                    0.265331          0.027114     0.149503        0.838966   0.257211             0.294210              0.529881  1.000000 -0.093591   0.074493 -0.780138 -0.307123
# pH                        -0.425858         -0.031915    -0.163748       -0.194133  -0.090439            -0.000618              0.002321 -0.093591  1.000000   0.155951  0.121432  0.099427
# sulphates                 -0.017143         -0.035728     0.062331       -0.026664   0.016763             0.059217              0.134562  0.074493  0.155951   1.000000 -0.017433  0.053678
# alcohol                   -0.120881          0.067718    -0.075729       -0.450631  -0.360189            -0.250104             -0.448892 -0.780138  0.121432  -0.017433  1.000000  0.435575
# quality                   -0.113663         -0.194723    -0.009209       -0.097577  -0.209934             0.008158             -0.174737 -0.307123  0.099427   0.053678  0.435575  1.000000

y = datasets["quality"]
x = datasets.drop(['quality', 'sulphates'], axis=1) # axis=1 컬럼 삭제할 때 필요함
print(y.unique())   # [6. 5. 7. 8. 4. 3. 9.]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
                    #  n_jobs = -1,
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

#3. 컴파일, 훈련

import time
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='merror',               # rmse, mae, logloss, error
          early_stopping_rounds=100,            # mlogloss, merror 
          )     

end = time.time() - start


#4. 평가, 예측

results = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("XGBClassifier : ", round(results, 4))
print("걸린시간 : ", round(end, 4),"초")
print("accuracy_score : ", round(acc, 4))

# XGBClassifier :  0.7286
# 걸린시간 :  16.8657 초
# accuracy_score :  0.7286
'''
