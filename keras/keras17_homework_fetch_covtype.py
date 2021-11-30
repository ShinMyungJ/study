from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.utils import to_categorical

#1. 데이터

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)        # (581012, 54)  (581012,)
print(np.unique(y))            # [1 2 3 4 5 6 7]
# y = to_categorical(y)
# print(y[:10])

import tensorflow as tf

#1-1. 데이터 변환 방식. sklearn

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)   # sparse=False인 이유 디폴트값인 True는 metrics 반환
                                    # array가 필요하기 때문에 False
y = ohe.fit_transform(y.reshape(-1,1)) # (-1, 정수) 세로로 세워지며 정수만큼의 값을 배열
                                       # (정수, -1) 가로로 세워지며 정수만큼의 값을 배열
print(y[:10])

'''
#1-2. 데이터 변환 방식. pandas
import pandas as pd
y = pd.get_dummies(y) # drop_first=True) <- 데이터프레임에서 필요로 하지않는 값(순번,..)을 삭제할 때, True를 사용. 첫째열 삭제
                      # dummy_na = True) <- Nan. 결측값도 인코딩에 포함함
'''

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, activation='linear', input_dim=54))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(50, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(15))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=54, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list  # batch_size 없을때 11581, 1일때 371847
                                                                                                                        # batch_size default값 = 32 (371847 / 11581 = 32.xxx)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# loss :  0.6238381266593933
# accurcy :  0.7341721057891846