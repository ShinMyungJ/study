from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)      # (442, 10) -> (442, 9) -> (442, 3, 3, 1)
print(y.shape)      # (442, )

print(datasets.feature_names)
# print(datasets.DESCR)

# data_x = pd.DataFrame(x)
# print(type(data_x))
# x = data_x.drop(['CHAS'], axis =1)
# print(type(x))

# 완료하시오!!
# train 0.7
# R2 0.8 이상

# cnn 맹그러

import pandas as pd
xx = pd.DataFrame(x, columns=datasets.feature_names)
print(type(xx))
print(xx)
# print(datasets.corr())

print(xx.corr())                    # 양의 상관관계 'weight' 가 양수, 음의 상관관계 'weight' 가 음수

# xx['sick'] = y

x = xx.drop(['s2'], axis =1)
# print(x.columns)
# print(x.shape)

x = x.to_numpy()

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(309, 3, 3, 1)
x_test = x_test.reshape(133, 3, 3, 1)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(3, 3, 1)))
model.summary()
model.add(Dropout(0.2))
model.add(Flatten())                                         
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k35_2_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=20, batch_size=8, validation_split=0.3, callbacks=[es, mcp])

# model = load_model("")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

# print(y_test.shape, y_predict.shape)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  4816.76416015625
# r2 스코어 :  0.22688962318817352