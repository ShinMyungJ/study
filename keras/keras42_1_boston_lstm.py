from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(datasets.feature_names)
# 506, 6, 13
print(y) 
print(x.shape)      # (506, 13) -> (506, 13, 1)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 13, 1)
x_test = x_test.reshape(x_test.shape[0], 13, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(50, input_length=13, input_dim=1))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k42_1_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.3, callbacks=[es, mcp])

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')


# model = load_model("")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
# print(y_test.shape, y_predict.shape)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
print(y_test.shape, y_predict.shape)

# CNN
# loss :  10.297506332397461
# r2 스코어 :  0.8753586643471483

# LSTM
# 걸린시간 :  15.798 초
# 5/5 [==============================] - 0s 1ms/step - loss: 16.4416
# loss :  16.4416446685791
# r2 스코어 :  0.8009898066880785