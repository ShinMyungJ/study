from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

datasets = load_breast_cancer()
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
input1 = Input(shape=(30,))
dense1 = Dense(5)(input1)
dense2 = Dense(20, activation='sigmoid')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)

model.save("./_save/keras25_7_save_model.h5")
# model = load_model('./_save/keras25_7_save_model.h5')

# model = Sequential()
# model.add(Dense(5, activation='linear', input_dim=30))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# 318/318 [==============================] - 0s 524us/step - loss: 0.0769 - accuracy: 0.9654 - val_loss: 0.0270 - val_accuracy: 1.0000
# loss :  0.07845936715602875
# accurcy :  0.9766082167625427