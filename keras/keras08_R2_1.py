## R2가 뭔지 찾아라!!!
'''
R2 값은 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것
1은 예측이 완벽한 경우고, 0은 훈련 세트의 출력값인 y_train의 평균으로만
예측하는 모델의 경우
R2 score : 실제 값의 분산 대비 예측값의 분산 비율, 결정계수
실제 값의 분산 대비 예측값의 분산 비율
R2=(Q-Qe/Q)=1-Qe/Q
Q = 전체 데이터의 편차들을 제곱하여 합한 값
Qe = 전체 데이터의 잔차들을 제곱하여 합한 값
잔차 : 종속변수와 독립변수와의 관계를 밝히는 통계모형에서 모형에
의하여 추정된 종속변수의 값과 실제 관찰된 종속변수 값과의 차이.
이 차이는 오차(error)로도 해석.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25])
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
                                                    # 데이터 파라미터 튜닝
#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(30))
model.add(Dense(200))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(500))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

'''
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
'''
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  8.097625732421875
# r2 스코어 :  0.7910829425941475
# loss :  4.448322772979736
# r2 스코어 :  0.6872273921424368