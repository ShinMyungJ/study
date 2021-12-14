# DNN은 2차원, RNN은 3차원, CNN은 4차원
# RNN에서 1,2,3,4,5,6,7
# batch_size = 행, timesteps = 열 

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

# input_shape = (행, 열, 몇개씩 자르는지!!!)
# reshape 할때 데이터와 순서가 바뀌면 안됨

x = x.reshape(4, 3, 1)            #   [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
# print(x)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, activation='linear', input_shape=(3,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')          # optimizer는 loss 값을 최적화 시킴
model.fit(x, y, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]])
print(result)

# [[7.9982038]]
# [[8.000103]]