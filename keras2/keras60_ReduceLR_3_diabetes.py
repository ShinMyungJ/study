from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

#1. 데이터

datasets = load_diabetes()
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
model.add(Dense(100, input_dim=10))
model.add(Dense(99, activation='relu'))
model.add(Dense(95))
model.add(Dense(90, activation='relu'))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

lr = 0.001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="mse", optimizer=optimizer)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k60_3_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.3, callbacks=[es, reducelr, mcp])

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
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('r2 : ', round(r2,4))
print("걸린시간 : ", round(end, 4), '초')

# CNN
# loss :  4816.76416015625
# r2 스코어 :  0.22688962318817352

# LSTM
# 걸린시간 :  25.006 초
# 5/5 [==============================] - 0s 997us/step - loss: 3712.3857
# loss :  3712.3857421875
# r2 스코어 :  0.40414682760888787

# ReduceLR
# learning rate :  0.001
# loss :  3262.8435
# r2 :  0.4763
# 걸린시간 :  18.1914 초