import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

#1. 데이터
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x.shape, y.shape)   # (569, 30) (569,)

print(y)
# print(y[:10])
print(np.unique(y))   # [0 1] 분류값. y의 라벨값이 어떤 것들이 있느냐

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=30))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])        
                    # 정확도 'accuracy' 결과값에 accuracy 출력됨. 훈련에 영향을 끼치지 않고 상황만 보여줌
                    # metrics : 2개 이상 list. accuracy 외에 추가적으로 사용할 수 있음.
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
print('loss : ', loss)                                                                    #                      지표로서 가치가 떨어짐
results = model.predict(x_test[:21])
print(y_test[:21])
print(results)

# loss :  [0.33236247301101685, 0.8947368264198303]

# Sigmoid 정의
# Sigmoid 함수는 모든 실수 입력 값을 0보다 크고 1보다 작은 미분 가능한
# 수로 변환하는 특징을 가짐
# Sigmoid S(x) = 1/1+e^-x = e^x/e^x+1
# 0< S(x) < 1
# 이렇게 하면 결과값 0과 1로 한정 시킬 수 있음
# sigmoid의 반환 값은 확률 형태이기 때문에 결과를 확률로 해석할 때 유용함
# 이진분류 결과값은 0~1 사이이며 결과값이 0 or 1이 아님
# 정해진 라벨값만 나오며 0.5를 기준으로 정해줌
# 이진분류는 activation = 'Sigmoid', loss = 'binaryCrossEntropy' '만' 쓴다

# 참고
# hyper parameter tuning은 알아서 할 것
# 결과값 or (hidden layer & 결과값)에 sigmoid를 사용해도 됨
# sigmoid를 사용하여 결과값 예측율을 많이 올려놓음
 