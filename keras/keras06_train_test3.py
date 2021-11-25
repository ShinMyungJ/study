from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=66) # 랜덤난수 사용하여 고정값으로 훈련
print(x_test)       # [ 8 93  4  5 52 41  0 73 88 68]
print(y_test)

#2. 모델생성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(200))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([101])
print('101의 예측값 : ', result)

# loss :  1.2288985544728348e-06
# 101의 예측값 :  [[102.00066]]