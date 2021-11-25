## R2가 뭔지 찾아라!!!
'''
R2 값은 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것
1은 예측이 완벽한 경우고, 0은 훈련 세트의 출력값인 y_train의 평균으로만
예측하는 모델의 경우
R2 score : 실제 값의 분산 대비 예측값의 분산 비율, 결정계수
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
model.fit(x, y, epochs=200, batch_size=1)


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