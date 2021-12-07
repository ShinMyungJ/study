from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=30))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dropout(0.2))
model.add(Dense(20, activation='sigmoid'))
# model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k28_3_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

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
318/318 [==============================] - 0s 575us/step - loss: 0.2442 - accuracy: 0.9119 - val_loss: 0.4895 - val_accuracy: 0.8250
loss :  0.3191918134689331
accurcy :  0.871345043182373

# MinMax
318/318 [==============================] - 0s 556us/step - loss: 0.0706 - accuracy: 0.9748 - val_loss: 0.0337 - val_accuracy: 0.9750
loss :  0.08240047842264175
accurcy :  0.9824561476707458

# Standard
318/318 [==============================] - 0s 568us/step - loss: 0.0510 - accuracy: 0.9843 - val_loss: 0.0586 - val_accuracy: 0.9750
loss :  0.04126803204417229
accurcy :  0.988304078578949

# Robust
318/318 [==============================] - 0s 568us/step - loss: 0.0437 - accuracy: 0.9874 - val_loss: 0.0661 - val_accuracy: 0.9750
loss :  0.06147442013025284
accurcy :  0.9766082167625427

# MaxAbs
318/318 [==============================] - 0s 568us/step - loss: 0.0878 - accuracy: 0.9528 - val_loss: 0.0323 - val_accuracy: 0.9875
loss :  0.08423151820898056
accurcy :  0.9766082167625427
'''

# ModelCheckPoint 사용해도 큰 차이는 없음
# loss :  0.0957830473780632
# accurcy :  0.9766082167625427

# Dropout을 모두 사용한 경우 성능이 약간 떨어짐
# loss :  0.1140415370464325
# accurcy :  0.9649122953414917

# Dropout을 한번만 사용한 경우 성능이 약간 떨어짐
# loss :  0.10662207007408142
# accurcy :  0.9473684430122375