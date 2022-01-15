import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)      # x = (150,4) y = (150,1)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)
# print(y)
# print(np.unique(y))     #   [0, 1, 2]

import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
print(y)
print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(5, activation='linear', input_dim=4))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dense(20, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()

#3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

model.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("result : ", result)
print("accuracy_score : ", acc)