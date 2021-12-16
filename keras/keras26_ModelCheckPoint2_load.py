from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(65))
# model.add(Dense(30))
# model.add(Dense(15))
# model.add(Dense(5))
# model.add(Dense(1))
# model.summary()



#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')                               # weight를 load 해줄때, compile까지 명시해줘야 한다

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True,
#                       filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, callbacks=[es, mcp])
# end = time.time() - start
# print("=================================================")
# print(hist.history['val_loss'])
# print("=================================================")

# print("걸린시간 : ", round(end, 3), '초')

# model.save("./_save/keras26_1_save_model.h5")

model = load_model('./_ModelCheckPoint/keras26_1_mcp.hdf5')
# model = load_model('./_save/keras26_1_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  35.21707534790039
# r2 스코어 :  0.5786570847910619

# loss :  36.061798095703125
# r2 스코어 :  0.5685507144497828

