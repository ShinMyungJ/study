from os import access
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)         # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)           # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/255.    # 순서가 바뀌지 않음
x_test = x_test.reshape(10000, 28, 28, 1)/255.

print(x_train.shape)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(128, kernel_size=(2,2), padding='valid', activation = 'relu', input_shape=(28, 28, 1)))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Flatten())                                         
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

lr = 0.001
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime
import time
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k61_2_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)
tb = TensorBoard(log_dir='../_save/_graph', histogram_freq = 0,
                 write_graph=True, write_images=True)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.25, callbacks=[es, reduce_lr, mcp, tb])
end = time.time() - start

# model = load_model("")

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('acc : ', round(acc,4))
print("걸린시간 : ", round(end, 4), '초')

###################### 시각화 ##########################
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()

########################
# learning rate :  0.001
# loss :  0.0563
# acc :  0.9928
# 걸린시간 :  617.4273 초
########################

# loss :  0.09230969846248627
# accurcy :  0.9832000136375427
# 걸린시간 :  932.6304 초

# 평가지표 acc
# 0.98

# learning rate :  0.01
# loss :  0.1278
# accurcy :  0.9649
# 걸린시간 :  155.2714 초

# learning rate :  0.001
# loss :  0.0549
# accurcy :  0.9881
# 걸린시간 :  169.3925 초

# learning rate :  0.0001
# loss :  0.0573
# accurcy :  0.9809
# 걸린시간 :  135.9617 초

# Learn
# learning rate :  0.001
# loss :  0.0449
# accurcy :  0.9905
# 걸린시간 :  156.7753 초

##################################################

# C:\Users\bitcamp>d:

# D:\>cd _save

# D:\_save>cd _graph

# D:\_save\_graph>dir/w
#  D 드라이브의 볼륨: 새 볼륨
#  볼륨 일련 번호: 1203-6D80

#  D:\_save\_graph 디렉터리

# [.]          [..]         [train]      [validation]
#                0개 파일                   0 바이트
#                4개 디렉터리  848,900,739,072 바이트 남음

# D:\_save\_graph>tensorboard --logdir=.

# Chrome 에서 http://localhost:6006/ or 127.0.0.1:6006 쓰면 사용가능

###################################################