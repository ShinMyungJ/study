from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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
input1 = Input(shape=(54,))
dense1 = Dense(10)(input1)
dense2 = Dense(50, activation='relu')(dense1)
dense3 = Dense(30)(dense2)
dense4 = Dense(15)(dense3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(10, activation='linear', input_dim=54))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='linear'))
# model.add(Dense(15))
# model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=54, validation_split=0.2, callbacks=[es])

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)