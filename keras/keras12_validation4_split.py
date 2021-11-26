from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8125, random_state=66)
# 13개, 3개


# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.23, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(15, input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3) # validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([100])
print("100의 예측값 : ", y_predict)

# loss :  6.2914236877986696e-06
# 100의 예측값 :  [[99.95652]]