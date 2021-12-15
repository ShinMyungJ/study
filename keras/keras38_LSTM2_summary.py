# LSTM(Long Short Term Memory) : 장단기 기억(Memory)
# LSTM은 RNN의 문제를 셀상태(Cell state)와 여러 개의 게이트(gate)를 가진 셀이라는 유닛을 통해 해결
# 이 유닛은 시퀀스 상 멀리 있는 요소를 잘 기억할 수 있도록 함
# 셀 상태는 기존 신경망의 은닉층이라고 생각할 수 있음
# 셀상태를 갱신하기 위해 기본적으로 3가지의 게이트가 필요함

# Forget, input, output 게이트
# Forget : 말 그대로 '과거 정보를 잊기'위한 게이트
#          이전 단계의 셀 상태를 얼마나 기억할 지 결정함. 0(모두 잊음)과 1(모두 기억) 사이의 값을 가짐
# input : '현재 정보를 기억하기' 위한 게이트
#          새로운 정보의 중요성에 따라 얼마나 반영할지 결정
# output : 최종 결과를 위한 게이트
#          셀 상태로부터 중요도에 따라 얼마나 출력할지 결정함

# 게이트는 가중치를 가진 은닉층으로 생각할 수 있음. 각 가중치는 sigmoid층에서 갱신되며 0과 1사이의 값을 가짐
# 이 값에 따라 입력되는 값을 조절하고, 오차에 의해 각 단계(time step)에서 갱신됨

# Activation Hyperbolic Tangent(tanh) Function

# sigmoid fuction을 보완하고자 나온 함수. 입력신호를 (−1,1) 사이의 값으로 normalization 해줌.
# 거의 모든 방면에서 sigmoid보다 성능이 좋음.
# 수식 : tanh(x) = e^x - e^-x / e^x + e^-x
#      d/dx tanh(x) = 1-tanh(x)^2


# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480            # params = dim(W)+dim(V)+dim(U) = n*n + k*n + n*m
# _________________________________________________________________     # n : dimension of hidden layer
# dense (Dense)                (None, 10)                110            # k : dimension of output layer
# _________________________________________________________________     # m : dimension of input layer
# dense_1 (Dense)              (None, 1)                 11             # input gate, forget gate, output gate


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])     # 아워너 80

print(x.shape, y.shape)    # (13, 3) (13,)

print(x)

# input_shape = (행, 열, 몇개씩 자르는지!!!)

x = x.reshape(13, 3, 1)            #   [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5],[6]]]
# print(x)

#2. 모델구성
model = Sequential()
model.add(LSTM(50, input_length=3, input_dim=1))
model.add(Dense(40, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
patience_num = 3000
model.compile(loss='mse', optimizer='adam')          # optimizer는 loss 값을 최적화 시킴
es = EarlyStopping(monitor='loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
model.fit(x, y, epochs=30000, batch_size=1, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[[77],[78],[79]]])
print(result)

# [[80.057236]]
# [[79.95282]]