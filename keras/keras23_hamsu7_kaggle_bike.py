import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터

path = "./_data/bike/"          # '.'은 현재 폴더 '..'은 이전단계

train = pd.read_csv(path + 'train.csv')
# print(train)                    # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
# print(test)                     # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')


# print(submit)                   # (6493, 2)

# print(type(train))                  # <class 'pandas.core.frame.DataFrame'>
# print(train.info())                 # 통상 type 이 object라면 string으로 간주하면 됨
# print(train.describe())             # mean 평균, std 표준편차
# print(train.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# print(train.head())                 # 위에서부터 5개 ()안에 숫자를 넣으면 그 수 많큼 나옴
# print(train.tail())               # 아래에서부터 5개

x = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)   # 횡으로 삭제할때가 디폴트, 열 삭제하려면 axis=1을 줘야함
test_file = test_file.drop(['datetime'], axis=1)

print(x.columns)                # column의 갯수가 8개로 줄음
print(x.shape)                  # (10886, 8)

y = train['count']
print(y)                       
print(y.shape)                  # (10886,)

print(submit_file.columns)

# print(test.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed'],
#       dtype='object')

# x_test = test.drop(['datetime'], axis=1)
# print(x_test)
# y_test = submit.drop(['datetime'], axis=1)
# print(y_test)

# 로그변환
y = np.log1p(y)             # log1p : y값을 로그변환 시킬때 1을 더해줌

# plt.plot(y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)


#2. 모델구성
input1 = Input(shape=(8,))
dense1 = Dense(50)(input1)
dense2 = Dense(25, activation='relu')(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=8))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(10))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 스코어 : ', r2)

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

# 871/871 [==============================] - 1s 579us/step - loss: 1.4565 - val_loss: 1.3557
# loss :  1.3814436197280884
# r2 스코어 :  0.2950216286472356
# RMSE :  1.1753482481331294