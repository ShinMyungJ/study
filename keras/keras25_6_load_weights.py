from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(65))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))
# model.summary()

# model.save("./_save/keras25_1_save_model.h5")
# model.save_weights("./_save/keras25_1_save_weights.h5")
# model.load_weights("./_save/keras25_1_save_weights.h5")
# loss :  9405.05859375
# r2 스코어 :  -111.52366952115946
# model.load_weights("./_save/keras25_3_save_weights.h5")


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                               # 컴파일까지 명시해줘야 한다
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=200, batch_size=13, validation_split=0.2)
# end = time.time() - start
# print("걸린시간 : ", round(end, 3), '초')

# model.save_weights("./_save/keras25_3_save_weights.h5")                             # fit 다음에 model.save하면 model과 weight까지 저장된다
model.load_weights("./_save/keras25_3_save_weights.h5")
# model.load_weights("./_save/keras25_1_save_weights.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)