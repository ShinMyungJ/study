from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

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
input1 = Input(shape=(10,))
dense1 = Dense(100)(input1)
dense2 = Dense(99, activation='relu')(dense1)
dense3 = Dense(95)(dense2)
dense4 = Dense(90, activation='relu')(dense3)
dense5 = Dense(30)(dense4)
dense6 = Dense(15)(dense5)
dense7 = Dense(5)(dense6)
output1 = Dense(1)(dense7)
model = Model(inputs=input1, outputs=output1)

model.save("./_save/keras25_6_save_model.h5")
# model = load_model('./_save/keras25_6_save_model.h5')

# model = Sequential()
# model.add(Dense(100, input_dim=10))
# model.add(Dense(99, activation='relu'))
# model.add(Dense(95))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(30))
# model.add(Dense(15))
# model.add(Dense(5))
# model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# 247/247 [==============================] - 0s 651us/step - loss: 1731.7236 - val_loss: 4780.4429
# loss :  3591.146484375
# r2 스코어 :  0.4236062778355636