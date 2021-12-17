#1. 데이터
import numpy as np
x1 = np.array([range(100), range(301, 401)])    # 삼성 저가, 고가
# x2 = np.array([range(101, 201), range(411, 511), range(100,200)])   # 미국선물 시가, 고가, 종가
x1 = np.transpose(x1)
# x2 = np.transpose(x2)

y1 = np.array(range(1001, 1101))
y2 = np.array(range(101,201))
y3 = np.array(range(401, 501))

print(x1.shape, y1.shape, y2.shape, y3.shape)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, y1, y2, y3, train_size=0.7, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)    #  (70,) (30,)
print(y2_train.shape, y2_test.shape)    #  (70,) (30,)
print(y3_train.shape, y3_test.shape)    #  (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input((2,))
dense1 = Dense(5, activation='relu', name = 'dense1')(input1)
dense2 = Dense(7, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(7, activation='relu', name = 'dense3')(dense2)
output1 = Dense(7, activation='relu', name = 'output1')(dense3)


#2-2 output 모델1
output21 = Dense(7)(output1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output 모델2
output31 = Dense(7)(output1)
output32 = Dense(11)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

#2-4 output 모델3
output41 = Dense(7)(output1)
output42 = Dense(11)(output41)
output43 = Dense(21)(output42)
output44 = Dense(11, activation='relu')(output43)
last_output3 = Dense(1)(output44)



model = Model(inputs= input1, outputs=[last_output1,last_output2,last_output3])

model.summary()

model.compile(loss="mse", optimizer="adam", metrics='mae')
hist = model.fit(x1_train, [y1_train, y2_train, y3_train], epochs=50, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x1_test ,[y1_test, y2_test, y3_test])
print('loss : ', loss)

y_predict = model.predict(x1_test)
y_predict = np.array(y_predict).reshape(3,30)          # (3,30,1) -> (3,30)
# print(y_predict.shape)
# print(y1_test.shape)

from sklearn.metrics import r2_score
r2 = r2_score([y1_test, y2_test, y3_test], y_predict)
print('r2 스코어 : ', r2)
# print(y_predict)