import numpy as np
from sklearn import metrics

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,6])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.00001
# optimizer = Adam(learning_rate=learning_rate)
# optimizer = Adadelta(learning_rate=learning_rate)
# optimizer = Adamax(learning_rate=learning_rate)
# optimizer = Adagrad(learning_rate=learning_rate)
# optimizer = RMSprop(learning_rate=learning_rate)
# optimizer = SGD(learning_rate=learning_rate)
optimizer = Nadam(learning_rate=learning_rate)

######## Adam(default) ########

# learning_rate=0.001,
# beta_1=0.9,
# beta_2=0.999,
# epsilon=1e-07,
# amsgrad=False,
# name="Adam",

###############################

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_predict = model.predict([11])

print('loss : ', round(loss, 4), 'lr : ', learning_rate, '결과물 : ', y_predict, 'optimizer : ', optimizer)

# loss :  3.2327 lr :  0.0001 결과물 :  [[11.006132]] optimizer :  <keras.optimizer_v2.adam.Adam object at 0x000002139000E970>
# loss :  3.0514 lr :  0.05 결과물 :  [[11.033909]] optimizer :  <keras.optimizer_v2.adadelta.Adadelta object at 0x000001F60865D970>
# loss :  2.9998 lr :  0.006 결과물 :  [[10.946761]] optimizer :  <keras.optimizer_v2.adamax.Adamax object at 0x000002AFA122D970>
# loss :  3.0259 lr :  0.007 결과물 :  [[11.066441]] optimizer :  <keras.optimizer_v2.adagrad.Adagrad object at 0x000001B1B008F970>
# loss :  3.2509 lr :  1e-05 결과물 :  [[11.075923]] optimizer :  <keras.optimizer_v2.rmsprop.RMSprop object at 0x000001F107B8E970>
# loss :  3.3046 lr :  1e-05 결과물 :  [[10.997776]] optimizer :  <keras.optimizer_v2.gradient_descent.SGD object at 0x0000019D1A54F970>
# loss :  3.2419 lr :  1e-05 결과물 :  [[11.055681]] optimizer :  <keras.optimizer_v2.nadam.Nadam object at 0x000002308B60E970>


