from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

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
dense1 = Dense(5)(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(3, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(5, activation='linear', input_dim=13))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# 99/99 [==============================] - 0s 624us/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.3488 - val_accuracy: 0.9600
# loss :  0.11525575071573257
# accurcy :  0.9814814925193787