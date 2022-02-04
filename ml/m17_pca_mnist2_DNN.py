# 실습
# 아까 4가지로 모델을 맹그러봐
# 784개 DNN으로 만든거와 비교!!!

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

scaler =MinMaxScaler()   
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)

# 0.95 = n_components=154
# 0.99 : n_components=331
# 0.999 : n_components=486
# 1.0 : n_components=706, 713

pca = PCA(n_components=713)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

#####################################################
# 실습 
# pca를 통해 0.95 이상인 n_component가 몇개???????
# 0.95 : n_components=154
# 0.99 : n_components=331
# 0.999 : n_components=486
# 1.0 : n_components=706, 713
# np.argmax 써라
#####################################################

# pca_EVR = pca.explained_variance_ratio_
# # print(pca_EVR)
# # print(sum(pca_EVR))

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum>1) + 1)

# result_val = [0.95 ,0.99, 0.999, 1.0]
# for i in result_val:
#     print(i,np.argmax(cumsum>i) + 1)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()


#2. 모델구성

model = Sequential()
model.add(Dense(64, input_shape=(x_train.shape[1], )))
# model.add(Dense(64, input_shape=(784, )))                   # 1줄로 펴진 이미지로 변환, 784개 컬럼으로 판단
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'm17_2_', datetime, '_', filename])
es = EarlyStopping(monitor='accuracy', patience=20, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="accuracy", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=16, validation_split=0.3, callbacks=[es, mcp])
end = time.time() - start

# model = load_model("")

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])
print("걸린시간 : ",round(end, 4), '초')

# 1. 나의 최고의 DNN
# time = 508.5025 초
# acc =  0.9490000009536743


# 2. 나의 최고의 CNN
# loss :  0.09230969846248627
# acc :  0.9832000136375427
# time :  932.6304 초

# 3. PCA 0.95
# accurcy :  0.9690999984741211
# 걸린시간 :  926.1452 초

# 4. PCA 0.99
# loss :  0.1548701971769333
# time = 913.6436 초
# acc = 0.9684000015258789

# 5. PCA 0.999
# loss :  0.21999746561050415
# time = 1245.2889 초
# acc = 0.9621000289916992

# 6. PCA 1.0
# loss :  0.2078629434108734
# time = 993.9652 초
# acc = 0.9646999835968018
