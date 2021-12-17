#1. 데이터
import numpy as np
x1 = np.array([range(100), range(301, 401)])    # 삼성 저가, 고가
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])   # 미국선물 시가, 고가, 종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001, 1101))

print(x1.shape, x2.shape, y.shape)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)   #  (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input((2,))
dense1 = Dense(5, activation='relu', name = 'dense1')(input1)
dense2 = Dense(7, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(7, activation='relu', name = 'dense3')(dense2)
output1 = Dense(7, activation='relu', name = 'output1')(dense3)

#2-2 모델2
input2 = Input((3,))
dense11 = Dense(10, activation='relu', name = 'dense11')(input2)
dense12 = Dense(10, activation='relu', name = 'dense12')(dense11)
dense13 = Dense(10, activation='relu', name = 'dense13')(dense12)
dense14 = Dense(10, activation='relu', name = 'dense14')(dense13)
output2 = Dense(5, activation='relu', name = 'output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2])            #(None, 12)
# print(merge1.shape)
merge1 = Concatenate(axis=-1)([output1, output2])      #(None, 12)
print(merge1.shape)
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

model.compile(loss="mse", optimizer="adam")
hist = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

y_predict = model.predict([x1_test,x2_test])

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
print(y_predict)