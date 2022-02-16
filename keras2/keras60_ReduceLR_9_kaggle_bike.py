from math import expm1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Reshape
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 
y = train['count']
y = np.log1p(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=x.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
import time

lr = 0.001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="mse", optimizer=optimizer)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k60_9_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.3, callbacks=[es, reducelr, mcp])

end = time.time() - start

#4. 평가, 예측
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
rmse = RMSE(y_test,y_pred)

print('RMSE : ', rmse)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('r2 : ', round(r2,4))
print("걸린시간 : ", round(end, 4), '초')
y_pred = np.expm1(y_pred)
print(y_pred[:5])

# CNN
# loss: 1.422896385192871
# r2 스코어: 0.28490085907862783
# RMSLE :  1.1928520594650784

# LSTM
# loss: 1.4208415746688843
# r2 스코어: 0.30242391214475783
# RMSLE :  1.191990571885143

# Conv1D
# 걸린시간 :  95.25 초
# 103/103 [==============================] - 0s 450us/step - loss: 1.3884
# loss :  1.3883616924285889
# r2 : 0.31837018804411465
# RMSLE :  1.1782876239351439

# RMSE :  1.1768680577139243
# learning rate :  0.001
# loss :  1.385
# r2 :  0.32
# 걸린시간 :  657.6155 초
# [[ 70.543564]
#  [391.04007 ]
#  [ 80.509346]
#  [ 46.675518]
#  [ 75.22818 ]]