#           삼성                    키움
# 1. 월     종가                    종가        0.1
# 2. 화     거래량                  거래량      0.2
# 3. 수     시가                    시가        0.7
# 소스 파일과 Weight 값

# 데이터셋 건들지 마라

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Input
from sklearn.model_selection import train_test_split


#1. 데이터

path = "./_data/stock/"
ss = pd.read_csv(path +"삼성전자.csv", thousands=',', encoding='cp949')
ss = ss.drop(range(20, 1120), axis=0)                                  # 물적분할 (893, 1120)
ki = pd.read_csv(path + "키움증권.csv", thousands=',', encoding='cp949')
ki = ki.drop(range(20, 1060), axis=0)
# ki = ki.drop(range(893, 1120), axis=0)

ss = ss.loc[::-1].reset_index(drop=True)
ki = ki.loc[::-1].reset_index(drop=True)
# y_ss = ss['종가']
# y_ki = ki['종가']

print(ss.columns)     # (1060, 17)
                      # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
                      # '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], dtype='object')
print(ki.columns)     # (1160, 17)
                      # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
                      #'금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], dtype='object')
# print(ss.info())
# print(ss.isnull().sum())    #(거래량 3, 금액(백만) 3)
# print(ki.isnull().sum())  

x_ss = ss.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ss = np.array(x_ss)
x_ki = ki.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ki = np.array(x_ki)
# x_ss = x_ss.tail(20)
# y_ss = y_ss.tail(20)
print(x_ss[:10])




def split_xy3(dataset, time_steps, y_column):                     # size : 몇개로 나눌 것인가
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(x_ss, 5, 2)
x2, y2 = split_xy3(x_ki, 5, 2)

print(x1, "\n", y1)
print(x2, "\n", y2)

print(x1.shape, y1.shape)     # (15, 5, 3) (15, 2)
print(x2.shape, y2.shape)     # (15, 5, 3) (15, 2)

# print(x_ss)
# print(x_ss.shape, x_ki.shape)   # (head(n), 3) (head(n), 3)
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)  #(10, 5, 4) (5, 5, 4)  <- (10, 5, 3) (5, 5, 3)
print(x2_train.shape, x2_test.shape)  #(10, 5, 4) (5, 5, 4)
print(y1_train.shape, y1_test.shape)  #(10, 1) (5, 1)
print(y2_train.shape, y2_test.shape)  #(10, 1) (5, 1)

# #2-1 모델1
# input1 = Input((5,3))
# dense1 = LSTM(10, activation='relu', name = 'dense1')(input1)
# dense2 = Dense(15, activation='relu', name = 'dense2')(dense1)
# dense3 = Dense(7, activation='relu', name = 'dense3')(dense2)
# output1 = Dense(7, activation='relu', name = 'output1')(dense3)

# #2-2 모델2
# input2 = Input((5,3))
# dense11 = LSTM(10, activation='relu', name = 'dense11')(input2)
# dense12 = Dense(30, activation='relu', name = 'dense12')(dense11)
# dense13 = Dense(25, activation='relu', name = 'dense13')(dense12)
# dense14 = Dense(10, activation='relu', name = 'dense14')(dense13)
# output2 = Dense(5, activation='relu', name = 'output2')(dense14)

# from tensorflow.keras.layers import concatenate, Concatenate
# # merge1 = concatenate([output1, output2])            # (None, 12)
# # print(merge1.shape)
# merge1 = Concatenate()([output1, output2])            # (None, 12)
# # print(merge1.shape)

# #2-3 output 모델1
# output21 = Dense(7)(merge1)
# output22 = Dense(11)(output21)
# output23 = Dense(11, activation='relu')(output22)
# last_output1 = Dense(1)(output23)

# #2-4 output 모델2
# output31 = Dense(7)(merge1)
# output32 = Dense(11)(output31)
# output33 = Dense(21)(output32)
# output34 = Dense(11, activation='relu')(output33)
# last_output2 = Dense(1)(output34)

# model = Model(inputs=[input1, input2], outputs=[last_output1,last_output2])

# model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics='mae')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# # print(datetime)

# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
# model_path = "".join([filepath, 'test_stock_', datetime, '_', filename])
# es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

# start = time.time()
# hist = model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.3, callbacks=[es, mcp])
# end = time.time() - start
# print("걸린시간 : ", round(end, 3), '초')

# model.save("./_save/test_stock1_save_model.h5")
# model = load_model('./_ModelCheckPoint/test_stock_1219_2118_0037-21398376.0000.hdf5')
model = load_model('./_save/test_stock1_save_model.h5')


#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss)
result1, result2 = model.predict([x1, x2])
print('삼성전자 12/20 종가 : ', result1[-1],'원')
print('키움증권 12/20 종가 : ', result2[-1],'원')
# print(y1_pred)
# print(y2_pred)

