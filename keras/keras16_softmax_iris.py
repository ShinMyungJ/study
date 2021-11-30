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
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=4))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
print('accurcy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

# 다중분류
# 고차함수보다 직선을 긋는 방식이 더 좋으며, 버릴 값은 버린다
# 과적합을 피하기 위해 고차함수 쓰지 않는 것이 좋음
# output 값 3개 이상
# 다중 분류에서 Activation ='softmax', loss = categorical_CrossEntropy를 사용
# 모델의 마지막 노드 갯수 = label의 갯수

# softmax function
# softmax function은 자신의 가중 합뿐만 아니라, 다른 출력 노드들의 가중 합도 고려
# 즉, 범주의 수만큼의 차원을 갖는 입력벡터를 받아서 확률(요소의 합이 1)로 변환해줌
# ex) [0.2 0.3 0.5] [0.1 0.1 0.8] [0.4 0.3 0.3] -> 총합은 1

# one-hot encoding
# 원핫(One-Hot) 인코딩이라는 말처럼 이 기술은 데이터를 수많은 0과 한개의 1의 값으로
# 데이터를 구별하는 인코딩
# 출력값이 one-hot encoding 된 결과로 나오고 실측 결과와의 비교시에도
# 실측 결과는 one-hot encoding 형태로 구성됨
# ex) [[0 0 1]
#      [0 1 0]
#      [1 0 0]]
# 네트윅 레이어 구성시 마지막에 Dense(3, activation='softmax')로 3개의 클래스 가각 별로
# positive 확률 값이 나오게 됨
# 위 네트윅 출력값과 실축값의 오차값을 계산한다
# one-hot encoding을 지나면 input y=(n, 1) -> y=(n, 3)으로 변환됨

# 정리
# 1. 다중 분류 데이터를 보면 라벨의 갯수부터 확인
# 2. 라벨은 one-hot encoding
# 3. activation은 softmax
# 4. loss는 categorical_CrossEntropy
# 5. 최종 output의 node 갯수와 라벨의 갯수 맞춤