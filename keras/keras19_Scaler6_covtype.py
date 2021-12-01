from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = Sequential()
model.add(Dense(10, activation='linear', input_dim=54))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(50, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(15))
model.add(Dense(7, activation='softmax'))

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


'''
# 결과

# 그냥
6026/6026 [==============================] - 4s 688us/step - loss: 0.6605 - accuracy: 0.7129 - val_loss: 0.6588 - val_accuracy: 0.7173
loss :  0.6546290516853333
accurcy :  0.7174189686775208

# MinMax
6026/6026 [==============================] - 4s 637us/step - loss: 0.6386 - accuracy: 0.7208 - val_loss: 0.6362 - val_accuracy: 0.7254
loss :  0.6383407115936279
accurcy :  0.7196966409683228

# Standard
6026/6026 [==============================] - 4s 657us/step - loss: 0.6365 - accuracy: 0.7222 - val_loss: 0.6359 - val_accuracy: 0.7207
loss :  0.634918212890625
accurcy :  0.7247968912124634

# Rubust
6026/6026 [==============================] - 4s 654us/step - loss: 0.6348 - accuracy: 0.7223 - val_loss: 0.6311 - val_accuracy: 0.7256
loss :  0.6335583329200745
accurcy :  0.7221062183380127

# MaxAbs
6026/6026 [==============================] - 4s 670us/step - loss: 0.6396 - accuracy: 0.7196 - val_loss: 0.6375 - val_accuracy: 0.7231
loss :  0.636986255645752
accurcy :  0.7233626246452332

'''