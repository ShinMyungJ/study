from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=54))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(15))
# model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k28_6_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=100, batch_size=54, validation_split=0.2, callbacks=[es, mcp])

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
6026/6026 [==============================] - 5s 850us/step - loss: 0.5346 - accuracy: 0.7686 - val_loss: 0.5343 - val_accuracy: 0.7710
loss :  0.5243490934371948
accurcy :  0.7735565304756165

# MinMax
6026/6026 [==============================] - 4s 668us/step - loss: 0.4486 - accuracy: 0.8093 - val_loss: 0.4460 - val_accuracy: 0.8104
loss :  0.44336816668510437
accurcy :  0.8129187822341919

# Standard
6026/6026 [==============================] - 4s 620us/step - loss: 0.4502 - accuracy: 0.8087 - val_loss: 0.4496 - val_accuracy: 0.8100
loss :  0.44817522168159485
accurcy :  0.8112435936927795

# Robust
6026/6026 [==============================] - 4s 644us/step - loss: 0.4305 - accuracy: 0.8201 - val_loss: 0.4328 - val_accuracy: 0.8204
loss :  0.4305780827999115
accurcy :  0.8219203352928162

# MaxAbs
6026/6026 [==============================] - 4s 636us/step - loss: 0.4646 - accuracy: 0.7996 - val_loss: 0.4605 - val_accuracy: 0.8041
loss :  0.45904219150543213
accurcy :  0.801863431930542
'''
# ModelCheckPoint 사용했을 경우 비교적 낮은 loss값과 좋은 r2스코어가 나옴
# loss :  0.4287237524986267
# accurcy :  0.8229472637176514

# Dropout을 모두 사용한 경우 성능이 떨어짐
# loss :  0.5973713397979736
# accurcy :  0.7417041659355164

# Dropout을 두번만 사용한 경우 성능이 약간 떨어짐
#loss :  0.5012425184249878
# accurcy :  0.7820360064506531