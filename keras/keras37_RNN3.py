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

x = x.reshape(4, 3, 1)            #(batch, timesteps, feature) #   [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
# print(x)

#2. 모델구성
model = Sequential()                                                
# model.add(SimpleRNN(10, activation='linear', input_shape=(3,1)))          # SimpleRNN(units, activation default=tanh, input_shape=(timesteps,feature)
# model.add(SimpleRNN(units=10, activation='linear', input_shape=(3,1)))      # hyperbolic tangent(tanh)가 무엇인가?
model.add(SimpleRNN(10, input_length=3, input_dim=1))      # hyperbolic tangent(tanh)가 무엇인가?

model.add(Dense(10, activation='relu'))
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