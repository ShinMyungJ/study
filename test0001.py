'''
print("Hello World")

a = 0.1
b = 0.2
print(a + b)
'''

#import tensorflow as tf
#print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5],[2,4,6,8,6]])
x = np.transpose(x)
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer ='adam')

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[6,4]])
print('[6, 12]의 예측값 : ', result)
