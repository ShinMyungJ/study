# 맹그러!!!

import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)

print(np.unique(y_train))
# print("LDA 전 : ", x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print("LDA 전 : ", x_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=25)
lda = LinearDiscriminantAnalysis(n_components=4)          # (최대 피쳐 수 or y의 라벨의 수) - 1 보다 크게 n_component를 넣을 수 없음. 여기는 y값이 2개라서 1만 가능.
# x = pca.fit_transform(x)
# x_train = lda.fit_transform(x_train, y_train)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA 후 : ", x_train.shape)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
# model.fit(x_train, y_train, eval_metric='error')
model.fit(x_train, y_train, eval_metric='merror')
# model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# mnist
# [0 1 2 3 4 5 6 7 8 9]
# LDA 전 :  (60000, 28, 28)
# LDA 후 :  (60000, 9)
# 결과 :  0.9163

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 8)
# 결과 :  0.909

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 7)
# 결과 :  0.8915

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 6)
# 결과 :  0.8676

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 5)
# 결과 :  0.842

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 4)
# 결과 :  0.8282