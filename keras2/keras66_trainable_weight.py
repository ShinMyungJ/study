import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)
print("====================================================")
print(model.trainable_weights)
print("====================================================")

print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

# model.trainable=False

print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, batch_size=1, epochs=100)

'''
model.summary()
## layer1
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.55600935, -1.1611063 ,  0.7191298 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.]
                              (input_dim이 1개에서 3개로 연산)                                                                      (bias 3개)                                                       bias의 초기값 
## layer2
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) # 2번째 layer
array([[ 0.18737733, -0.23796391],
       [ 1.0758598 ,  0.26983893],
       [-0.04089153,  0.48064458]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
## layer3
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy= array([[-1.1171147],
       [-0.8005173]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''
'''
Non-trainable params: 29  >> 가중치 갱신을 하지 않겠다.
'''


