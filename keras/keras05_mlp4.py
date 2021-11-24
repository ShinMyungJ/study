import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터                   # 데이터 정제과정, 중요!
x = np.array([range(10)])
print(x)
x = np.transpose(x)
print(x.shape)              # (10, 3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.5, 1.4, 1.3],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.transpose(y)
print(y.shape)

# [9] 결과값 예측

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(150))
model.add(Dense(40))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs = 200, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([9])
print('[9]의 예측값 : ', y_predict)

# loss :  0.013821432366967201
# [9]의 예측값 :  [[9.985922 1.290771 1.080203]]