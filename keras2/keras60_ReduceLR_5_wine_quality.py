# csv 파일

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, PolynomialFeatures, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#1. 데이터 
path = '../_data/winequlity/'   
datasets = pd.read_csv(path+'winequality-white.csv',
                       index_col=None, sep=';', header=0, dtype=float)
print(datasets.columns)
print(datasets.info())
print(datasets.shape)   # (4898, 12)
print(datasets.corr())
print(datasets.describe())

datasets = datasets.values      # numpy로 변환
print(type(datasets))
print(datasets.shape)

x = datasets[:, :-1]
y = datasets[:, -1]
print(x.shape)
print(y.shape)

from tensorflow.keras.utils import to_categorical
y = pd.get_dummies(y)
print(y.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, x_test.shape)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)

#2 모델구성
model = Sequential()
model.add(Dense(60, input_dim=x.shape[1]))
model.add(Dense(42, activation='relu'))
model.add(Dense(35))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(5, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

lr = 0.001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['acc']) # metrics=['accuracy'] 영향을 미치지 않는다
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
import time

date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k60_5_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.3, callbacks=[es, reducelr, mcp])

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('acc : ', round(acc,4))
print("걸린시간 : ", round(end, 4), '초')

# ReduceLR
# learning rate :  0.001
# loss :  1.043
# acc :  0.5459
# 걸린시간 :  196.235 초

