from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)      # (581012, 54) -> (581012, 6, 3, 3)
print(y.shape)      # (581012, )

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y.reshape(-1,1))

# print(datasets.feature_names)
# print(datasets.DESCR)

# import pandas as pd
# xx = pd.DataFrame(x, columns=datasets.feature_names)
# print(type(xx))
# print(xx)
# print(datasets.corr())

# print(xx.corr())                    # 양의 상관관계 'weight' 가 양수, 음의 상관관계 'weight' 가 음수

# xx['cov'] = y

# x = xx.drop(['s2'], axis =1)
# print(x.columns)
# print(x.shape)

# x = x.to_numpy()

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)

x_train = x_train.reshape(406708, 6, 3, 3)
x_test = x_test.reshape(174304, 6, 3, 3)
print(x_train.shape, x_test.shape)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(6, 3, 3)))
model.summary()
model.add(Dropout(0.2))
model.add(Flatten())                                         
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k35_6_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=54, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# loss :  4816.76416015625
# r2 스코어 :  0.22688962318817352