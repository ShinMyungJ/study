import pandas as pd
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, Reshape
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

# 실습!!!

#1. 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)         # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)           # (10000, 28, 28) (10000,)

# print(np.unique(y_train))                   # [0 1 2 3 4 5 6 7 8 9]
      
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

n = x_train.shape[0]
x_train_reshape = x_train.reshape(n,-1)
print(x_train_reshape.shape)

x_train = scaler.fit_transform(x_train_reshape)
m = x_test.shape[0]
x_test = x_test.reshape(m,-1)
x_test = scaler.transform(x_test.reshape(m,-1))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성

model = Sequential()
model.add(Conv1D(10, 2, input_shape=(x_train.shape[1],1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k44_9_', datetime, '_', filename])
es = EarlyStopping(monitor='accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="accuracy", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.3, callbacks=[es, mcp])

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

# model = load_model("")

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

# DNN
# loss :  0.4538135826587677
# accurcy :  0.8411999940872192

# LSTM
# loss :  1.0381516218185425
# accurcy :  0.5985999703407288

# Conv1D
# 걸린시간 :  230.614 초
# 313/313 [==============================] - 0s 1ms/step - loss: 0.4578 - accuracy: 0.8595
# loss :  0.45777568221092224
# accurcy :  0.859499990940094