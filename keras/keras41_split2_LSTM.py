import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


#1. 데이터
a = range(1, 101)
x_predict = np.array(range(96, 106))
print(x_predict)

size = 5

def split_x(dataset, size):                     # size : 몇개로 나눌 것인가
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# print(bbb.shape)

x = bbb[:, :4]
y = bbb[:, 4]

print(x, y)
print(x.shape, y.shape)  #(96, 4) (96, )

pred = split_x(x_predict, 5)
# print(pred)
x_pred = pred[:, :4]
print(x_pred.shape)
x_pred = x_pred.reshape(6,4,1)
print(x_pred.shape)

x = x.reshape(x.shape[0],4,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 66) 

# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = RobustScaler()
# # scaler = MaxAbsScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)

# x_train = x_train.reshape(x_train.shape[0],4,1)
# x_test = x_test.reshape(x_test.shape[0],4,1)

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
patience_num = 200
model.compile(loss='mse', optimizer='adam')          # optimizer는 loss 값을 최적화 시킴
es = EarlyStopping(monitor='loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(x_pred)
print(result)


# [[ 99.84699 ]
#  [100.71543 ]
#  [101.5592  ]
#  [102.377525]
#  [103.16981 ]
#  [103.935555]]