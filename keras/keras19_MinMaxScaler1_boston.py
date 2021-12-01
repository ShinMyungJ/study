###################################################################
# 각각의 Scaler의 특성과 정의 정리해놓을것!!!
###################################################################
# 1) StandardScaler : 특성들의 평균을 0, 분산을 1로 스케일링하는 것(특성들을 정규분포로 만드는 것)
#    최솟값과 최댓값의 크기를 제한하지 않기 때문에, 어떤 알고리즘에서는 문제가 있을 수 있음.
#    이상치에 매우 민감함
#    회귀보다 분류에 유용함
# 2) MinMaxScaler : Min-Max Normalization 이라고 불리며, 특성들을 특정 범위(주로[0,1])로 스케일링 하는 것.
#    가장 작은 값은 0, 가장 큰 값은 1로 변환되므로, 모든 특성들은 [0,1] 범위를 갖게 됨
#    이상치에 매우 민감함
#    분류보다 회귀에 유용함
# 3) MaxAbsScaler : 각 특성의 절대값이 0과 1 사이가 되도록 스케일링 함
#    즉, 모든 값은 -1과 1 사이로 표현되며, 데이터가 양수일 경우 MinMaxScaler와 같음
#    이상치에 매우 민감함
# 4) RobustScaler : 평균과 분산 대신에 중간 값과 사분위 값을 사용함
#    중간 값은 정렬시 중간에 있는 값을 의미하고, 사분위값은 1/4, 3/4에 위치한 값을 의미함
#    이상치 영향을 최소화 할 수 있음

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
model.add(Dense(30, activation = 'relu'))
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

# MinMax(회기)
283/283 [==============================] - 0s 524us/step - loss: 25.9883 - val_loss: 38.4885
loss :  19.64570426940918
r2 스코어 :  0.7622077900793867

# Standard(분류)
283/283 [==============================] - 0s 542us/step - loss: 26.7449 - val_loss: 41.6062
loss :  18.43268394470215
r2 스코어 :  0.7768902213013731

# Robust(이상치 영향 최소화)
283/283 [==============================] - 0s 559us/step - loss: 29.0476 - val_loss: 37.7693
loss :  19.603111267089844
r2 스코어 :  0.7627233545804679

# MaxAbs(회기, 양수인경우 MinMax와 같음)
283/283 [==============================] - 0s 537us/step - loss: 27.8096 - val_loss: 36.7153
loss :  16.342443466186523
r2 스코어 :  0.802190537388543



# relu를 사용한 결과

# 그냥
283/283 [==============================] - 0s 558us/step - loss: 17.6541 - val_loss: 39.0207
loss :  12.606269836425781
r2 스코어 :  0.8474132894789859

# MinMax
283/283 [==============================] - 0s 540us/step - loss: 5.8708 - val_loss: 22.6893
loss :  8.897183418273926
r2 스코어 :  0.8923081946590172

# Standard
283/283 [==============================] - 0s 539us/step - loss: 7.0008 - val_loss: 13.8371
loss :  9.011690139770508
r2 스코어 :  0.890922226775651

# Robust
283/283 [==============================] - 0s 672us/step - loss: 7.8800 - val_loss: 17.4005
loss :  10.796603202819824
r2 스코어 :  0.8693175628255357

# MaxAbs
283/283 [==============================] - 0s 556us/step - loss: 12.6778 - val_loss: 37.2488
loss :  11.364872932434082
r2 스코어 :  0.8624392179929752
'''