# DNN은 2차원, RNN은 3차원, CNN은 4차원
# RNN에서 1,2,3,4,5,6,7
# (4, 3, 1) -> (batch_size, timesteps, features)
# batch_size = 행, timesteps = 열, features = 몇개씩 자를건지

#            input        output         비고
# DNN        2차원         2차원           x
# RNN        3차원         2차원           x
# Conv2D     4차원         4차원        Flatten
# Conv1D     3차원

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)    # (4, 3) (4,)

print(x)

# input_shape = (행, 열, 몇개씩 자르는지!!!)
# reshape 할때 데이터와 순서가 바뀌면 안됨

x = x.reshape(4, 3, 1)            #   [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
# print(x)

#2. 모델구성
model = Sequential()                                                # Total = recurrent_weights + input_weights + biases
model.add(SimpleRNN(20, activation='linear', input_shape=(3,1)))    # (num_units * num_units) + (num_features * num_units) + (1 * num_units)
model.add(Dense(10, activation='relu'))                             # (num_features + num_units) * num_units + num_units
# model.add(Dense(16, activation='relu'))                           # (unit 개수 * unit 개수) + (input_dim(feature) 수 * unit 개수) + (1 * unit 개수)
# model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')          # optimizer는 loss 값을 최적화 시킴
# model.fit(x, y, epochs=300, batch_size=1)

# #4. 평가, 예측
# loss = model.evaluate(x, y)
# result = model.predict([[[5],[6],[7]]])
# print(result)

# [[7.9982038]]
# [[8.00024]]