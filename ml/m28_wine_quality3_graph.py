import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

#1. 데이터 
path = '../_data/winequlity/'   
datasets = pd.read_csv(path+'winequality-white.csv',
                       index_col=None, sep=';', header=0, dtype=float)
# print(datasets.columns)
# print(datasets.info())
# print(datasets.shape)   # (4898, 12)
# print(datasets.corr())
# print(datasets.describe())

# datasets = datasets.values      # numpy로 변환
# print(type(datasets))
# print(datasets.shape)

# x = datasets[:, :-1]
# y = datasets[:, -1]
# print(x.shape)
# print(y.shape)
# print('라벨 : ', np.unique(y, return_counts=True))
# 라벨 :  (array([ 3.,  4.,   5.,   6.,   7.,   8.,   9.]),
#          array([20,  163, 1457, 2198,  880,  175,   5], dtype=int64))

############################ 아웃라이어 확인 #####################################
# 해봐!!
# pd.value_counts 쓰지말고
# groupby 쓰고, count() 써라
# plt.bar 로 그려라 quality

import matplotlib.pyplot as plt


# def boxplot_vis(data, target_name):
#     plt.figure(figsize=(10, 10))
#     for col_idx in range(len(data.columns)):
#         # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
#         plt.subplot(6, 2, col_idx+1)
#         # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
#         plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
#         # 그래프 타이틀: feature name
#         plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 8)
#     # plt.savefig('../figure/boxplot_' + target_name + '.png')
#     plt.show()
# boxplot_vis(datasets,'white_wine')

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

# 이상치 제거한 데이터셋
prep = remove_outlier(datasets)     # 이상치를 NaN으로
print(prep.head(30))
print(datasets.head(30))

# xx = prep.groupby('quality').count()
# print(datasets.groupby('quality').count())
# print(datasets.columns.to_list())
# label = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# index = np.arange(len(label))
# print(index)

# a = prep.isna().sum()
# a.plot.bar()
# plt.show()
# g1 = prep.groupby([ "quality"]).count()
# g2 = datasets.groupby([ "quality"]).count()
# print(g1)
# print(g2)
# g3 = g2 - g1
# g3.plot.bar()
# # p = pd.DataFrame({'count' : prep.groupby( [ "quality"] ).size()})
# # print(p)
# # p.plot(kind='bar', rot=0)

# # count_data = datasets.groupby('quality')['quality'].count()
# # print(count_data)

# # count_data.plot.bar()
# # plt.bar(count_data.index, count_data)
# plt.show()

"""
# 목표변수 할당
prep['target'] = 0

# 결측치(이상치 처리된 데이터) 확인
a = prep.isnull().sum()
print(a)
'''
fixed acidity           146
volatile acidity        186
citric acid             270
residual sugar            7
chlorides               208
free sulfur dioxide      50
total sulfur dioxide     19
density                   5
pH                       75
sulphates               124
alcohol                   0
quality                 200
'''

# 이상치 포함 데이터(이상치 처리 후 NaN) 삭제
prep.dropna(axis = 0, how = 'any', inplace = True)
print(f"이상치 포함된 데이터 비율: {round((len(datasets) - len(prep))*100/len(datasets), 2)}%")
#이상치 포함된 데이터 비율: 21.58%

x = prep.drop('quality', axis=1)
y = prep['quality']

print(x.shape, y.shape) #(3841, 12) (3841,)

############################ 아웃라이어 처리 #####################################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
# scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
                    #  n_jobs = -1,
                     n_estimators = 5000,
                     learning_rate = 0.1,
                     max_depth = 8,
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
score = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print("model.score : ", score)
print("accuracy score : ", accuracy_score(y_test, y_predict))
print("f1 score : ", f1_score(y_test, y_predict, average='macro'))
print("f1 score : ", f1_score(y_test, y_predict, average='micro'))

# XGBClassifier :  0.7286
# 걸린시간 :  16.8657 초
# accuracy_score :  0.7286
"""
