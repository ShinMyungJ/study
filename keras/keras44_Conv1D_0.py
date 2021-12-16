import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten

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
# model.add(SimpleRNN(20, activation='linear', input_shape=(3,1)))    # (num_units * num_units) + (num_features * num_units) + (1 * num_units)
# model.add(Bidirectional(SimpleRNN(10), input_shape=(3,1)))                             # (num_features + num_units) * num_units + num_units
model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Dense(10, activation='relu'))                             # (num_features + num_units) * num_units + num_units
model.add(Flatten())
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
# [[8.00024]]