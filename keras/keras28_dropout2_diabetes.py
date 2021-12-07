from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

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
model.add(Dense(100, input_dim=10))
model.add(Dropout(0.4))
model.add(Dense(99, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(95))
# model.add(Dropout(0.2))
model.add(Dense(90))
# model.add(Dropout(0.2))
model.add(Dense(30))
# model.add(Dropout(0.2))
model.add(Dense(15))
# model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k28_2_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


'''
# 결과

# 그냥
247/247 [==============================] - 0s 641us/step - loss: 1693.1047 - val_loss: 5005.4390
loss :  3266.333251953125
r2 스코어 :  0.47574009726370003

# MinMax
247/247 [==============================] - 0s 649us/step - loss: 2067.9333 - val_loss: 4093.6340
loss :  3186.283447265625
r2 스코어 :  0.4885884876812623

# Standard
247/247 [==============================] - 0s 662us/step - loss: 1087.1880 - val_loss: 4922.7319
loss :  3305.08154296875
r2 스코어 :  0.46952087243744134

# Robust
247/247 [==============================] - 0s 637us/step - loss: 1212.7340 - val_loss: 6813.3784
loss :  3070.999755859375
r2 스코어 :  0.5070919290388716

# MaxAbs
247/247 [==============================] - 0s 655us/step - loss: 1380.8918 - val_loss: 5730.0195
loss :  3586.58642578125
r2 스코어 :  0.4243382129846396
'''

# ModelCheckPoint 사용해도 큰 차이는 없음
# loss :  3260.634765625
# r2 스코어 :  0.4766547242352527

# Dropout을 모두 사용한 경우 성능이 약간 상승
# loss :  3522.298828125
# r2 스코어 :  0.4346565326031392

# Dropout을 한번 사용한 경우 성능이 약간 상승
# loss :  3546.777099609375
# r2 스코어 :  0.43072772247195246