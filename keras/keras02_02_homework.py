# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
# 요 데이터를 훈련해서 최소의 loss를 만들어보자.

#2. 모델
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(160))
model.add(Dense(2000))
model.add(Dense(74))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=30, batch_size=1)                  # 30번 돌리니

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4])
print('4의 예측값 : ', result)

# loss :  6.159683834994212e-05
# 4의 예측값 :  [[4.000191]]
# 30번 돌렸을때