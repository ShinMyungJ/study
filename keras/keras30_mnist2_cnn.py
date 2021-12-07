from os import access
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)         # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)           # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)    # 순서가 바뀌지 않음
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# #2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1)))
model.add(Conv2D(5, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation='relu'))
model.add(Flatten())                                         
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k30_2_', datetime, '_', filename])
es = EarlyStopping(monitor='accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="accuracy", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
hist = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.3, callbacks=[es, mcp])

# model = load_model("")

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

# loss :  0.07771419733762741
# accurcy :  0.9807000160217285

# 평가지표 acc
# 0.98