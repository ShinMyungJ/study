import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(150, 150),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=5,
    class_mode='binary',
    shuffle=True
)       # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'    
)       # Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001B7BC3D4F70>

# from sklearn.datasets import load_boston
# dataset = load_boston()
# print(dataset)

# print(xy_train[31])       # 마지막 batch
# print(xy_train[0][0])
# print(xy_train[0][1])
# print(xy_train[0][2])             # error
print(xy_train[0][0].shape, xy_train[0][1].shape)         # (5, 150, 150, 3), (5,)   # 흑백은 알아서 찾아라

print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
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
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k47_2_generator_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=xy_test,
                    validation_steps=4, callbacks=[es, mcp]
                    )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 점심때 그래프 그려보아요

import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(acc,)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# plt.plot(epochs, loss, 'r--', label="loss")
# plt.plot(epochs, val_loss, 'r:', label="loss")
# plt.plot(epochs, acc, 'b--', label="acc")
# plt.plot(epochs, val_acc, 'b:', label="val_acc")

# 걸린시간 :  100.577 초
# loss :  0.524624228477478
# val_loss :  0.2718330919742584
# acc :  0.768750011920929
# val_acc :  0.949999988079071