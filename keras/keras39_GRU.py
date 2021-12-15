import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])     # 아워너 80

print(x.shape, y.shape)    # (13, 3) (13,)

print(x)

# input_shape = (행, 열, 몇개씩 자르는지!!!)

x = x.reshape(13, 3, 1)            #   [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
# print(x)

#2. 모델구성
model = Sequential()
model.add(GRU(50, input_length=3, input_dim=1))
# model.add(LSTM(50, input_length=3, input_dim=1))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
patience_num = 150
model.compile(loss='mse', optimizer='adam')          # optimizer는 loss 값을 최적화 시킴
es = EarlyStopping(monitor='loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
model.fit(x, y, epochs=300, batch_size=1, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[[77],[78],[79]]])
print(result)
