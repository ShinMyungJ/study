# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])                     # 50000
print(randidx)                              # [46616 39506  2172 ...   840  4805 17478]     
print(np.min(randidx), np.max(randidx))     # 2 49999

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)        # (50000, 32, 32, 3)
print(y_augmented.shape)        # (50000, 1)

import time
start_time = time.time()
print("시작!!!")
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                #  save_to_dir='../_temp/'
                                 ).next()[0]
end_time = time.time() - start_time

print("걸린시간 : ", round(end_time, 3), '초')

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train)
print(x_train.shape, y_train.shape) # (100000, 32, 32, 3) (100000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# x_train = x_train.reshape(-1,28,28,3)

# # x_data = train_datagen.flow(
# #     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),   # x
# #     np.zeros(augment_size),                                                    # y
# #     batch_size=augment_size,
# #     shuffle=False
# # ).next()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)      # (100000, 10) (10000, 10)

#2. 모델
model = Sequential()
model.add(Conv2D(30, kernel_size=(3,3), strides=2, padding='same',
                 input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(2))
model.add(Conv2D(15,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(7,(2,2), activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k50_2_', datetime, '_', filename])
es = EarlyStopping(monitor='acc', patience=20, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="acc", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.3, callbacks=[es, mcp])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
scores = model.evaluate(x_test, y_test)
print("%s: %.2f" %(model.metrics_names[0], scores[0]))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

predict = model.predict(x_test)
print(predict[:3])

# loss: 1.13
# acc: 62.21%

