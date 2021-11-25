from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
                                                    # 데이터 파라미터 튜닝
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=3)
end = time.time() - start
print("걸린시간 : ", end)

"""
verbose
0: 없다         2.6387081146240234
1: 다           3.7968904972076416
2: loss까지     3.0413568019866943
3~: epoch만     2.796344518661499
'''

'''
#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict(x)

'''
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
'''
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2 스코어 : ', r2)

# loss :  0.3800048828125
# r2 스코어 :  0.8099974724335596
"""