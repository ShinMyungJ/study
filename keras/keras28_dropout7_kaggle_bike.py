import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10))
# model.add(Dropout(0.2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k28_7_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 스코어 : ', r2)

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

'''
# 결과

# 그냥
871/871 [==============================] - 0s 538us/step - loss: 1.5325 - val_loss: 1.4312
loss :  1.4280275106430054
r2 스코어 :  0.27124892561517366
RMSE :  1.1950010085321567

# MinMax
871/871 [==============================] - 0s 522us/step - loss: 1.4746 - val_loss: 1.3454
loss :  1.3822821378707886
r2 스코어 :  0.2945936166288661
RMSE :  1.1757049873351113

# Standard
871/871 [==============================] - 0s 522us/step - loss: 1.4620 - val_loss: 1.3328
loss :  1.3636717796325684
r2 스코어 :  0.3040909735074777
RMSE :  1.1677635160059705

# Robust
871/871 [==============================] - 0s 517us/step - loss: 1.4323 - val_loss: 1.3362
loss :  1.3479743003845215
r2 스코어 :  0.3121017129903306
RMSE :  1.1610228891443835

# MaxAbs
871/871 [==============================] - 0s 533us/step - loss: 1.4592 - val_loss: 1.3195
loss :  1.3963655233383179
r2 스코어 :  0.2874066945655933
RMSE :  1.1816790521761782
'''
# ModelCheckPoint 사용했을 경우 비교적 낮은 loss값과 좋은 r2스코어, RMSLE가 나옴
# loss :  1.3450175523757935
# r2 스코어 :  0.3136106675399446
# RMSE :  1.1597487962983075

# Dropout을 모두 사용한 경우 비슷한 결과가 나옴
# loss :  1.3831522464752197
# r2 스코어 :  0.29414962341347617
# RMSE :  1.1760749321898762

# Dropout을 두번만 사용한 경우 비슷한 결과가 나옴
# loss :  1.3487968444824219
# r2 스코어 :  0.3116818251633914
# RMSE :  1.1613771748130506