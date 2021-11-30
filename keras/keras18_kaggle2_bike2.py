### 과제1. 평균값과 중위값의 차이를 정리하라 ###
# 평균값 : N 개의 변량을 모두 더하여 그 개수로 나누어 놓은 숫자. 또는 산술평균.
# 중위값 : N 개의 값을 크기 순으로 늘어놓았을 때 가장 가운데에 있는 숫자. 표본들의 격차가 클때 사용.

# log 값의 문제점 : 0이 나오면 안되기 때문에 +1 해줌

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터

path = "./_data/bike/"          # '.'은 현재 폴더 '..'은 이전단계

train_raw = pd.read_csv(path + 'train.csv')
# print(train)                    # (10886, 12)
test_file = pd.read_csv(path + 'test.csv')
# print(test)                     # (6493, 9)
submit_file = pd.read_csv(path + 'sampleSubmission.csv')
# print(submit)                   # (6493, 2)

train = train_raw.copy()
train['datetime'] = pd.to_datetime(train['datetime'])     # datetime 컬럼 데이터 타입을 datetime으로 변경
train.dtypes

print(train.isnull().sum())        # 결측치 확인

train['year'] = train['datetime'].dt.year           # 날짜 데이터를 정수 형태로 변환
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
train['dayofweek'] = train['datetime'].dt.dayofweek  # 요일 데이터 - 일요일은 0

#print(train.describe())

count_q1 = np.percentile(train['count'], 25)
print(count_q1)
count_q3 = np.percentile(train['count'], 75)
print(count_q3)
# IQR = Q3 - Q1
count_IQR = count_q3 - count_q1
count_IQR
train_clean = train[(train['count'] >= (count_q1 - (1.5 * count_IQR))) & (train['count'] <= (count_q3 + (1.5 * count_IQR)))]
train_wo_outliers = train_clean[np.abs(train_clean["count"] - train_clean["count"].mean()) <= (3*train_clean["count"].std())]
def to_integer(datetime):
      return 10000 * datetime.year + 100 * datetime.month + datetime.day
datetime_int = train_wo_outliers['datetime'].apply(lambda x: to_integer(x))
train_wo_outliers['datetime'] = pd.Series(datetime_int)

print(train.info())

'''
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


#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=20, batch_size=4, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print('r2 스코어 : ', r2)

rmse = RMSE(y_test, y_pred)
print("RMSE : ", rmse)

# loss :  23809.03125
# r2 스코어 :  0.24673393686610068
# RMSE :  154.30175497542078

# 로그변환 후
# loss :  1.4543837308883667
# r2 스코어 :  0.2577987458594312
# RMSE :  1.2059783455683972


######################제출용 제작##############################
results = model.predict(test_file)

submit_file['count'] = results

# print(submit_file[:10])

submit_file.to_csv(path + "aaaaaa.csv", index=False)

#3.10659
'''