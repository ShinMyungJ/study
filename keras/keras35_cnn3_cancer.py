# File 3. Breast Cancer Data
from sklearn.utils import validation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 데이터
# Data load
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569, )

# DATA 개요(x)
    # print(datasets.feature_names)
    # print(datasets.DESCR)
    # data_x = pd.DataFrame(x)
    # print(type(data_x))
    # x = data_x.drop(['CHAS'], axis=1)
    # print(type(x))
    # x_train = x_train.drop('CHAS',axis = 1)
    # print(x_train)

# # PANDAS 형변환
xx = pd.DataFrame(x, columns = datasets.feature_names)
# print(type(xx))
# print(xx)
# print(datasets.corr() )
# print(xx.corr())          # 양의 상관관계 'weight'가 양수, 음의 상관관계 'weight'가 음수
xx['discrimination'] = y      # 유방암의 양성 '판별' <= y

x = xx.drop(['worst fractal dimension'], axis =1)       # 상관관계가 0에 가까운 'wfd' 를 제거
# print(x.columns)
# print(x.shape)
#
# # x를 넘파이형 데이터로 변환
x = x.to_numpy()

# # plot확인
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()


# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

# 전처리 4종
# a. MinMaxScaler
scaler = MinMaxScaler()
# b. MinMaxScaler
# scaler = StandardScaler()
# c. RobustScaler
# scaler = RobustScaler()
# d. MaxAbsScaler
# scaler = MaxAbsScaler()
# Scaler fit & transform
scaler.fit(x_train)
x_train = scaler.transform(x_train)           # train은 minmaxscaler가 됨
x_test = scaler.transform(x_test)            # 여기까지 x에 대한 전처리
                                    # y는 타겟일 뿐이기에 안 한다(필기 참고_ 쌤's 군대사격 예시든 부분)

# print(x_train.shape)#(398, 30)
# print(x_test.shape) #(171, 30)
x_train = x_train.reshape(398, 3, 2, 5)
x_test = x_test.reshape(171, 3, 2, 5)


# 2. 모델링
# Sequential Ver Set

model = Sequential()

model.add(Conv2D(10, kernel_size=(2,2), input_shape=(3, 2, 5)))
model.summary()
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=3, verbose=1, validation_split=0.2, callbacks=[es])



# 4. 평가, 예측

# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


# 결과 report
# 고정값 : 0.7trainsize, 66th_randomstate, 20patience, 100epochs, 3batch, 0.2val
# loss :  2.8929258988341644e-09 accuracy :  1.0