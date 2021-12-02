import numpy as np

#1. 데이터

x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])
print(x.shape, y.shape)    # (3, 100) (1, 100)
x = np.transpose(x)
y = np.transpose(y)
# print(x.shape, y.shape)    # (100, 3) (10, 1)
# x = x.reshape(1, 10, 10, 3)   # 4차원 데이터. 전형적인 이미지의 데이터의 형태
# print(x.shape, y.shape)    # (1, 10, 10, 3) (10, 1)


#2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)   # 모델구성은 중간부터도 가능함, Dense 또한 바로 위가 아니더라도 가능은 함

# model = Sequential()                         # Sequential 한계점 : 2차원 한정. 3차원에서 부터는 shape이 더 필요함
# # model.add(Dense(10, input_dim=3))          # (100, 3)  ->  (N, 3)
# model.add(Dense(10, input_shape=(3,)))       # input_shape 사용하면 모든 차원 가능 맨앞 차원 빼고 넣어주면 됨
# model.add(Dense(9))                          #               ex) (1, 10, 10, 3) -> input_shape = (10, 10, 3)
# model.add(Dense(8))     
# model.add(Dense(1))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 9)                 99
# _________________________________________________________________
# dense_2 (Dense)              (None, 8)                 80
# _________________________________________________________________
# dense_3 (Dense)              (None, 7)                 63
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 8
# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________


