import numpy as np
import time

# np.save('./_save_npy/keras47_4_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras47_4_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras47_4_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras47_4_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_4_train_x.npy')
y_train = np.load('./_save_npy/keras47_4_train_y.npy')
x_test = np.load('./_save_npy/keras47_4_test_x.npy')
y_test = np.load('./_save_npy/keras47_4_test_y.npy')

print(x_train)
print(x_train.shape)

#2. 모델 구성하시오!!!

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(2))
model.add(Conv2D(16, (2,2)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k47_6_load_', datetime, '_', filename])

es = EarlyStopping(monitor='val_accuracy', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_accuracy', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=4, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_acc[-1])

