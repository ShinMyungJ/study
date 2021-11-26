from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)
print(x.shape, y.shape)
print(datasets.feature_names)
print(datasets.DESCR)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# R2
# 0.62 이상이상
# train 0.6 ~ 0.8

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(14))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)        # r2 보다 loss가 더 신뢰성 있다
print('r2 스코어 : ', r2)

# loss :  1969.408935546875
# r2 스코어 :  0.6303878386265211