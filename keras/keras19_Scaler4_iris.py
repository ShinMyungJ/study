from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

datasets = load_iris()
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

#2. 모델 구성

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=4))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

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
84/84 [==============================] - 0s 656us/step - loss: 0.1118 - accuracy: 0.9762 - val_loss: 0.0166 - val_accuracy: 1.0000
loss :  0.1063741147518158
accurcy :  0.9333333373069763

# MinMax
84/84 [==============================] - 0s 645us/step - loss: 0.1028 - accuracy: 0.9643 - val_loss: 0.0326 - val_accuracy: 1.0000
loss :  0.12553265690803528
accurcy :  0.9333333373069763

# Standard
84/84 [==============================] - 0s 642us/step - loss: 0.0645 - accuracy: 0.9643 - val_loss: 0.0219 - val_accuracy: 1.0000
loss :  0.09181448817253113
accurcy :  0.9555555582046509

# Robust
84/84 [==============================] - 0s 647us/step - loss: 0.0660 - accuracy: 0.9524 - val_loss: 0.0596 - val_accuracy: 0.9524
loss :  0.1883831024169922
accurcy :  0.9333333373069763

# MaxAbs
84/84 [==============================] - 0s 657us/step - loss: 0.1032 - accuracy: 0.9762 - val_loss: 0.0193 - val_accuracy: 1.0000
loss :  0.11873067170381546
accurcy :  0.9333333373069763

# relu를 사용한 결과

# 그냥
84/84 [==============================] - 0s 635us/step - loss: 0.0987 - accuracy: 0.9643 - val_loss: 0.0141 - val_accuracy: 1.0000
loss :  0.06353810429573059
accurcy :  0.9777777791023254

# MinMax
84/84 [==============================] - 0s 667us/step - loss: 0.0507 - accuracy: 0.9881 - val_loss: 0.0246 - val_accuracy: 1.0000
loss :  0.17684966325759888
accurcy :  0.9111111164093018

# Standard
84/84 [==============================] - 0s 675us/step - loss: 0.0560 - accuracy: 0.9762 - val_loss: 0.1632 - val_accuracy: 0.9524
loss :  0.18186378479003906
accurcy :  0.9333333373069763

# Robust
84/84 [==============================] - 0s 660us/step - loss: 0.0433 - accuracy: 0.9881 - val_loss: 0.0694 - val_accuracy: 0.9524
loss :  0.16363933682441711
accurcy :  0.9555555582046509

# MaxAbs
84/84 [==============================] - 0s 672us/step - loss: 0.1008 - accuracy: 0.9405 - val_loss: 0.0224 - val_accuracy: 1.0000
loss :  0.1539481282234192
accurcy :  0.9333333373069763
'''