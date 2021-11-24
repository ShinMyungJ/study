# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])
# 요 데이터를 훈련해서 최소의 loss를 만들어보자.

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=4000, batch_size=1)                  # 4000번 돌리니

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

# loss :  0.38000011444091797
# 6의 예측값 :  [[5.700002]]
# 4000번 돌렸을때