# [실습 / 과제]
# [실습] cifar10을 넣어서 완성할것!!

# vgg trainable : True, False
# Flatten / GlobalAveragepooling
# autokeras
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,GlobalAvgPool2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size=augment_size) # randint - 랜덤한 정수값을 뽑는다
print(x_train.shape[0]) # 50000

print(randidx) # [32882 21036 43516 ... 48177 49866 51437]
print(np.min(randidx), np.max(randidx)) # 0 ~ 59996

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape) # (50000, 28, 28)
print(y_augmented.shape) # (50000,)

x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir = '../_temp/'
                                 ).next()[0]

print(x_augmented)
print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train)
print(x_train.shape) # augmented 와 합쳐진 x_train mnist 값
print(y_test.shape) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_test.shape) # (10000, 28, 28, 1)

# 2. 모델구성

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (32, 32, 3))
# vgg16.trainable = False     # 가중치를 동결시킨다.

model = Sequential()
model.add(vgg19)
model.add(GlobalAvgPool2D())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(learning_rate = learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'cifar100_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=20, mode = 'auto', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, filepath = model_path)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto', verbose = 1, factor= 0.5)

start = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size = 32, validation_split = 0.25, callbacks = [es, reduce_lr])#, mcp])
end = time.time()- start

# model = load_model('./_ModelCheckPoint/mnist_1207_1843_0015-0.0745.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('learning_late : ', learning_rate)
print('accuracy : ', loss[1])
print("걸린시간 : ", round(end, 3), '초')
y_predict = model.predict(x_test)

'''
flatten
loss:  60.25337600708008
learning_late :  0.0001
accuracy :  0.8783000111579895
걸린시간 :  3829.628 초
'''