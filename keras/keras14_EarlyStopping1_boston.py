# EarlyStopping을 사용하여 모델 훈련을 멈췄을때,
# Best 성능을 가진 모델을 저장하는 것인가? Patience만큼 다음 모델이 저장되는 것인가?
# Best 성능 모델에서 Patience만큼 지난 모델이 저장된다.
# 그래서 callback함수를 이용하여야 한다.


from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(99))
model.add(Dense(95))
model.add(Dense(90))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping                  # 파라미터도 있으면 정리
es = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.2, callbacks=[es])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

print("=================================================")
print(hist)
print("=================================================")
print(hist.history)
print("=================================================")
print(hist.history['loss'])
print("=================================================")
print(hist.history['val_loss'])
print("=================================================")

import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')          # 선을 긋는다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()                  # 점자 형태
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()