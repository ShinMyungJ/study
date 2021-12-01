from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=13))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

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

'''
# 결과

# 그냥
99/99 [==============================] - 0s 622us/step - loss: 0.1451 - accuracy: 0.9495 - val_loss: 0.2842 - val_accuracy: 0.8800
loss :  0.15754136443138123
accurcy :  0.9444444179534912

# MinMax
99/99 [==============================] - 0s 635us/step - loss: 2.3017e-04 - accuracy: 1.0000 - val_loss: 0.3452 - val_accuracy: 0.9200
loss :  0.16895487904548645
accurcy :  0.9444444179534912

# Standard
99/99 [==============================] - 0s 673us/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 9.5367e-09 - val_accuracy: 1.0000
loss :  0.3968231678009033
accurcy :  0.9814814925193787

# Rubust
99/99 [==============================] - 0s 634us/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
loss :  0.37358835339546204
accurcy :  0.9814814925193787

# MaxAbs
99/99 [==============================] - 0s 624us/step - loss: 0.0038 - accuracy: 1.0000 - val_loss: 0.1565 - val_accuracy: 0.9600
loss :  0.1719604879617691
accurcy :  0.9629629850387573

'''