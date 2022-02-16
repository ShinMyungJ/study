from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)   # sparse=False인 이유 디폴트값인 True는 metrics 반환
                                    # array가 필요하기 때문에 False
y = ohe.fit_transform(y.reshape(-1,1))

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
model.add(Dense(128, activation='linear', input_dim=54))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

lr = 0.001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="mse", optimizer=optimizer, metrics=['acc'])
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

hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, callbacks=[es, reducelr, mcp])

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')


# model = load_model("")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
# print(y_test.shape, y_predict.shape)
loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('acc : ', round(acc,4))
print("걸린시간 : ", round(end, 4), '초')

# CNN
# 걸린시간 :  489.282 초
# 5447/5447 [==============================] - 4s 686us/step - loss: 0.4478 - accuracy: 0.8117
# loss :  0.44781821966171265
# accurcy :  0.8117427229881287

# LSTM
# 걸린시간 :  1285.607 초
# 5447/5447 [==============================] - 11s 2ms/step - loss: 0.7417 - accuracy: 0.6547
# loss:  0.7417375445365906
# accuracy :  0.6546780467033386

# ReduceLR
# learning rate :  0.001
# loss :  0.0186
# acc :  0.9198
# 걸린시간 :  3234.1847 초