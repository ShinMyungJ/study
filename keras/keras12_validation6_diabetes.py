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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# R2
# 0.62 이상
# train 0.6 ~ 0.8

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dense(77))
model.add(Dense(58))
model.add(Dense(44))
model.add(Dense(29))
model.add(Dense(14))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2 스코어 : ', r2)

# loss :  1908.4654541015625
# r2 스코어 :  0.641825535374549