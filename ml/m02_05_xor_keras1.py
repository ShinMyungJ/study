import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# AND 데이터 훈련
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC() # polynomial(다항식) : 곡선개념이 들어감 x^2-2x+3, 5xy+6.....
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)
results = model.evaluate(x_data, y_data)

print(x_data, "의 예측결과 : ", y_predict)
print("metrics_acc :", results[1])

# acc = accuracy_score(y_data, np.round(y_predict))
# print("accuracy_score : ", acc)


