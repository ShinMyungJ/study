from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(99))
model.add(Dense(60))
model.add(Dense(90))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# start = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=3, validation_split=0.2)
# end = time.time()
# print("걸린시간 : ", end) # v= 1 1637899557.6507597
                          # v= 3 1637899519.7166517

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  15.83908462524414
# r2 스코어 :  0.8104985728599116