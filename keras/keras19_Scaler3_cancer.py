from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=30))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)


'''
# 결과

# 그냥
318/318 [==============================] - 0s 528us/step - loss: 0.3114 - accuracy: 0.8962 - val_loss: 0.1533 - val_accuracy: 0.9125
loss :  0.3121068775653839
accurcy :  0.9181286692619324

# MinMax
318/318 [==============================] - 0s 580us/step - loss: 0.0744 - accuracy: 0.9780 - val_loss: 0.0463 - val_accuracy: 0.9750
loss :  0.059903908520936966
accurcy :  0.9824561476707458

# Standard
6/6 [==============================] - 0s 0s/step - loss: 0.0419 - accuracy: 0.9825
loss :  0.04186691343784332
accurcy :  0.9824561476707458

# Rubust
318/318 [==============================] - 0s 544us/step - loss: 0.0638 - accuracy: 0.9748 - val_loss: 0.0303 - val_accuracy: 0.9750
loss :  0.04804043471813202
accurcy :  0.9824561476707458

# MaxAbs
318/318 [==============================] - 0s 542us/step - loss: 0.0851 - accuracy: 0.9654 - val_loss: 0.0968 - val_accuracy: 0.9500
loss :  0.073243148624897
accurcy :  0.9766082167625427

'''