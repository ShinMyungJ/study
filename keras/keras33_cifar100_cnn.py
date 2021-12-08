from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train,return_counts=True))
'''
(50000, 32, 32, 3)
(50000, 1)
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
      dtype=int64))
'''
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#1 데이터
#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3])

scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(30, kernel_size=(3,3), strides=2, padding='same',
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(2))
model.add(Conv2D(15,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(7,(2,2), activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))

#model.summary()#3,153

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy']) # metrics=['accuracy'] 영향을 미치지 않는다

start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

############################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k33_cifar100_', datetime, '_', filename])
                   # ./_ModelCheckPoint/k33_cifar100_1206_1656_2500-0.3724.hdf5
#############################################################################

es = EarlyStopping(monitor='val_loss', patience=10, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = 32, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
#4 평가예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print("loss : ",loss[0])
print("accuracy : ",loss[1])