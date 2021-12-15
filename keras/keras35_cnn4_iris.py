import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

#1 데이터
datasets = load_iris()

x = datasets.data 
y = datasets.target
y = to_categorical(y)

print(x.shape)# (150, 4)
print(y.shape)# (150, 3)
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)
scaler = MinMaxScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(105, 2, 2, 1)
x_test = x_test.reshape(45, 2, 2, 1)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(2, 2, 1)))
model.add(Dropout(0.2))
model.add(Flatten())                                         
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation = 'softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'iris_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=50, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.3, callbacks = [es,mcp])
end = time.time()- start

#4. 평가, 예측

print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/iris_1206_2333_0050-0.0031.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

# loss:  0.09043966978788376
# accuracy :  0.9555555582046509