from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

# print(np.min(x), np.max(x))      # 0.0 711.0
# x = x/711.           # x. 으로 나누는 것 부동소수점으로 나눈다는 뜻. ex) 이미지 파일에 사용가능. x/255.
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

input1 = Input(shape=(13,))
dense1 = Dense(50)(input1)
dense2 = Dense(30, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(15)(drop1)
dense4 = Dense(8)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(15))
# model.add(Dense(8))
# model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# 283/283 [==============================] - 0s 549us/step - loss: 13.2942 - val_loss: 35.8075
# loss :  11.487203598022461
# r2 스코어 :  0.8609585300449373