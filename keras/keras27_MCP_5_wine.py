from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=13))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k27_5_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

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
99/99 [==============================] - 0s 636us/step - loss: 0.1361 - accuracy: 0.9495 - val_loss: 0.4580 - val_accuracy: 0.8000
loss :  0.20706646144390106
accurcy :  0.9074074029922485

# MinMax
99/99 [==============================] - 0s 620us/step - loss: 1.2510e-06 - accuracy: 1.0000 - val_loss: 0.0193 - val_accuracy: 1.0000
loss :  0.1591642200946808
accurcy :  0.9629629850387573

# Standard
99/99 [==============================] - 0s 626us/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 6.8618e-04 - val_accuracy: 1.0000
loss :  0.7050156593322754
accurcy :  0.9629629850387573

# Robust
99/99 [==============================] - 0s 620us/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 5.0589e-06 - val_accuracy: 1.0000
loss :  0.06395326554775238
accurcy :  0.9629629850387573

# MaxAbs
loss :  0.26385682821273804
accurcy :  0.8888888955116272
'''
# ModelCheckPoint 사용해도 큰 차이는 없음
# loss :  0.2180425524711609
# accurcy :  0.9444444179534912