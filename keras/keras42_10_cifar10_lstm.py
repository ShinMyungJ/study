import pandas as pd
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# 실습!!!

#1. 데이터

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

# print(x_train.shape)
# print(y_train.shape)
# print(np.unique(y_train,return_counts=True))
'''
(50000, 32, 32, 3)
(50000, 1)
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
      dtype=int64))
'''
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1)
print(x_train_reshape.shape)

x_train = scaler.fit_transform(x_train_reshape)
m = x_test.shape[0]
x_test = x_test.reshape(m,-1)
x_test = scaler.transform(x_test.reshape(m,-1))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성

model = Sequential()
model.add(LSTM(5, input_length=x_train.shape[1], input_dim=1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k42_10_', datetime, '_', filename])
es = EarlyStopping(monitor='accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="accuracy", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.2, callbacks=[es, mcp])

# model = load_model("")

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

# CNN
# loss :  0.9427627921104431
# accuracy :  0.6711000204086304

# DNN
# loss :  1.4445456266403198
# accuracy :  0.49000000953674316

# LSTM