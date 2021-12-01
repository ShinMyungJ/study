###################################################################
# 각각의 Scaler의 특성과 정의 정리해놓을것!!!
###################################################################

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

# print(np.min(x), np.max(x))      # 0.0 711.0
# x = x/711.           # x. 으로 나누는 것 부동소수점으로 나눈다는 뜻. ex) 이미지 파일에 사용가능. x/255.
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)



'''
# 결과

# 그냥
283/283 [==============================] - 0s 537us/step - loss: 30.5631 - val_loss: 35.7517
loss :  19.954885482788086
r2 스코어 :  0.7584654336981813

# MinMax
283/283 [==============================] - 0s 524us/step - loss: 25.9883 - val_loss: 38.4885
loss :  19.64570426940918
r2 스코어 :  0.7622077900793867

# Standard
283/283 [==============================] - 0s 542us/step - loss: 26.7449 - val_loss: 41.6062
loss :  18.43268394470215
r2 스코어 :  0.7768902213013731

# Rubust
283/283 [==============================] - 0s 559us/step - loss: 29.0476 - val_loss: 37.7693
loss :  19.603111267089844
r2 스코어 :  0.7627233545804679

# MaxAbs
283/283 [==============================] - 0s 537us/step - loss: 27.8096 - val_loss: 36.7153
loss :  16.342443466186523
r2 스코어 :  0.802190537388543

'''