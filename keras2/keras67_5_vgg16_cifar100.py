from pyparsing import col
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


# vgg trainable : True, False
# Flatten / GAP
# time도 명시해라!!!

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
n = x_train.shape[0]                           
x_train_reshape = x_train.reshape(n,-1)                
x_train_transe = scaler.fit_transform(x_train_reshape) 
x_train = x_train_transe.reshape(x_train.shape)    

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3))

# vgg16.summary()
vgg16.trainable = False     # 가중치를 동결시킨다!

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.trainable = False

########################### 2번에서 아래만 추가 ###############################

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
import time

lr = 0.00005
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics='accuracy')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k67_5_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, reducelr])#, mcp])

end = time.time() - start

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('accurcy : ', round(acc,4))
print("시간 : ", round(end, 4), '초')



# 출력결과
# 기존
# accuracy :  0.31

# VGG : False, Flatten
# learning rate :  5e-05
# loss :  2.5368
# accurcy :  0.3628
# 시간 :  655.1838 초

# VGG : False, GAP
# learning rate :  5e-05
# loss :  2.5425
# accurcy :  0.3608
# 시간 :  649.5443 초

# VGG : True, Flatten
# learning rate :  5e-05
# loss :  1.7928
# accurcy :  0.5431
# 시간 : 395.5591 초

# VGG : True, GAP
# learning rate :  5e-05
# loss :  1.7555
# accurcy :  0.5451
# 시간 :  336.1448 초


