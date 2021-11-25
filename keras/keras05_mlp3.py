# MLP(Multi-Layer Perceptron) 다층 퍼셉트론은 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태
# 퍼셉트론은 초기의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터                   # 데이터 정제과정, 중요!
x = np.array([range(10), range(21,31), range(201, 211)])
print(x)
x = np.transpose(x)
print(x.shape)              # (10, 3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.5, 1.4, 1.3],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.transpose(y)
print(y.shape)

# [[9, 30, 210]] 결과값 예측

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(30))
model.add(Dense(100))
model.add(Dense(15))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs = 300, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 : ', y_predict)

# loss :  0.016565844416618347
# [9, 30, 210]의 예측값 :  [[10.005939   1.3393663  1.0115778]]