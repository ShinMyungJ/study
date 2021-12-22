#           삼성                    키움
# 1. 월     종가                    종가        0.1
# 2. 화     거래량                  거래량      0.3
# 3. 수     시가                    시가        0.6
# 소스 파일과 Weight 값

# 데이터셋 건들지 마라

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터

path = "./_data/stock/"
ss = pd.read_csv(path +"삼성전자.csv", thousands=',', encoding='cp949')
ss = ss.drop(range(893, 1120), axis=0)                                  # 액면분할 (893, 1120)
ki = pd.read_csv(path + "키움증권.csv", thousands=',', encoding='cp949')
ki = ki.drop(range(893, 1060), axis=0)

ss = ss.loc[::-1].reset_index(drop=True)
ki = ki.loc[::-1].reset_index(drop=True)
# y_ss = ss['종가']
# y_ki = ki['종가']

# print(ss.corr())        # 시가 0.035651, 고가 0.065375, 저가 0.005420, 종가 0.026833, Unnamed: 6 -0.159778, 등락률 -0.154009, 거래량 1.000000, 금액(백만) 0.988491
                        # 신용비 0.040153 개인 0.491117 기관 -0.051554 외인(수량) -0.513440 외국계 -0.465529 프로그램 -0.437299 외인비 0.040693
# print(ki.corr())        


# print(ss.columns)     # (1060, 17)
                      # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
                      # '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], dtype='object')
# print(ki.columns)     # (1160, 17)
                      # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
                      #'금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], dtype='object')
# print(ss.info())
# print(ss.isnull().sum())    #(거래량 3, 금액(백만)  3)
# print(ki.isnull().sum())  

x_ss = ss.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ss = np.array(x_ss)
x_ki = ki.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ki = np.array(x_ki)
xx1 = ss.drop(['일자', '전일비', '시가', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
xx1 = np.array(xx1)
xx2 = ki.drop(['일자', '전일비', '시가', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
xx2 = np.array(xx2)
# print(x_ss.shape)       # (200, 4)


def split_xy3(dataset, time_steps, y_column):                     # size : 몇개로 나눌 것인가
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 1:]                       # [a,b]  a= 데이터의 위치, b = 칼럼의 위치
        tmp_y = dataset[x_end_number - 1:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(x_ss, 5, 4)
x2, y2 = split_xy3(x_ki, 5, 4)

print(x1.shape, y1.shape)     # (194, 5, 5) (194, 3)
print(x2.shape, y2.shape)     # (194, 5, 5) (194, 3)

def split_x(dataset, size):                     # size : 몇개로 나눌 것인가
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

pred1 = split_x(xx1, 5)
pred2 = split_x(xx2, 5)
print(pred1[-1])

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle=True, random_state=66)

x1_train  = x1_train.reshape(x1_train.shape[0],x1_train.shape[1]*x1_train.shape[2])
x1_test_x  = x1_test.reshape(x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1]*x2_train.shape[2])
x2_test  = x2_test.reshape(x2_test.shape[0],x2_test.shape[1]*x2_test.shape[2])
pred1  = pred1.reshape(pred1.shape[0],pred1.shape[1]*pred1.shape[2])
pred2  = pred2.reshape(pred2.shape[0],pred2.shape[1]*pred2.shape[2])


scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x2_test)
x2_train = scaler.fit_transform(x2_train)
x2_test_x = scaler.transform(x2_test)
pred1 = scaler.fit_transform(pred1)
pred2 = scaler.transform(pred2)

x1_train  = x1_train.reshape(x1_train.shape[0],5,3)
x1_test  = x1_test.reshape(x1_test_x.shape[0],5,3)

x2_train  = x2_train.reshape(x2_train.shape[0],5,3)
x2_test  = x2_test.reshape(x2_test.shape[0],5,3)

pred1  = pred1.reshape(pred1.shape[0],5,3)
pred2  = pred2.reshape(pred2.shape[0],5,3)

#2-1 모델1
input1 = Input((5,3))
dense1 = LSTM(10, activation='relu', name = 'dense1')(input1)
dense2 = Dense(64, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(32, activation='relu', name = 'dense3')(dense2)
output1 = Dense(16, name = 'output1')(dense3)

#2-2 모델2
input2 = Input((5,3))
dense11 = LSTM(10, activation='relu', name = 'dense11')(input2)
dense12 = Dense(64, activation='relu', name = 'dense12')(dense11)
dense13 = Dense(32, activation='relu', name = 'dense13')(dense12)
dense14 = Dense(16, activation='relu', name = 'dense14')(dense13)
output2 = Dense(8, name = 'output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])            # (None, 12)
# print(merge1.shape)
merge1 = Concatenate()([output1, output2])            # (None, 12)
# print(merge1.shape)

#2-3 output 모델1
output21 = Dense(32)(merge1)
output22 = Dense(16)(output21)
output23 = Dense(8, activation='relu')(output22)
last_output1 = Dense(4)(output23)

#2-4 output 모델2
output31 = Dense(32)(merge1)
output32 = Dense(24)(output31)
output33 = Dense(16)(output32)
output34 = Dense(8, activation='relu')(output33)
last_output2 = Dense(4)(output34)

model = Model(inputs=[input1, input2], outputs=[last_output1,last_output2])

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'test_stock_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist = model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=500, batch_size=4, validation_split=0.3, callbacks=[es])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

model = load_model('./_save/exam/3_78636_108945.h5')

#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss)
# pred1 = np.array(x_ss[-5:,1:]).reshape(1,5,6)
# pred2 = np.array(x_ki[-5:,1:]).reshape(1,5,6)
# print(pred1)
# print(pred2)

result1, result2 = model.predict([pred1, pred2])

ssp = result1[-1][-1]
kip = result2[-1][-1]

# model.save("./_save/exam/3_{0:.0f}_{1:.0f}.h5".format(ssp,kip))

print('삼성전자 12/22 시가 : ', f'{ssp:.0f}','원')
print('키움증권 12/22 시가 : ', f'{kip:.0f}','원')

# 삼성전자 12/22 시가 :  77564.75 원
# 키움증권 12/22 시가 :  108678.71 원

# 삼성전자 12/22 시가 :  77250.19 원
# 키움증권 12/22 시가 :  108673.66 원

# MaxAbsScaler
# 삼성전자 12/22 시가 :  84088.05 원
# 키움증권 12/22 시가 :  161724.69 원

# RobustScaler
# 삼성전자 12/22 시가 :  85106.805 원
# 키움증권 12/22 시가 :  192903.62 원

# MinMaxScaler
# 삼성전자 12/22 시가 :  89093.57 원
# 키움증권 12/22 시가 :  204731.11 원

# StandardScaler
# 삼성전자 12/22 시가 :  90040.17 원
# 키움증권 12/22 시가 :  185767.5 원
