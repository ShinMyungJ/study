import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,
              1.6,1.5,1.4,1.3]])
# x = x.reshape(2,10)           # 행과 열 바꾸기 // 의도와 다르게 배열됨, 이미지나 주가 데이터에는 사용하는 것이 좋음
x = np.swapaxes(x, 0, 1)
# x = np.transpose(x)
# x = x.T

y = np.array([11,12,13,14,15,16,17,18,19,20])
# print(x.shape)                # 행열확인
# print(y.shape)
# print(x)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3]])
print('[10, 1.3]의 예측값 : ', y_predict)

# loss :  0.48778676986694336
# [10, 1.3]의 예측값 :  [[19.693377]]