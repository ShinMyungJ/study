import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_diabetes()
# print(datasets.DESCR)      # x = (150,4) y = (150,1)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)
# print(y)
# print(np.unique(y))     #   [0, 1, 2]

import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
print(y)
print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


model1 = Perceptron()
model2 = LinearSVC()
model3 = SVC()
model4 = KNeighborsClassifier()
# model4 = KNeighborsRegressor()
model5 = LogisticRegression()
# model5 = LinearRegression()
model6 = DecisionTreeClassifier()
# model6 = DecisionTreeRegressor()
model7 = RandomForestClassifier()
# model7 = RandomForestRegressor()

#3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)
model5.fit(x_train, y_train)
model6.fit(x_train, y_train)
model7.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss[0])                                                                 #                      지표로서 가치가 떨어짐
# print('accurcy : ', loss[1])
result1 = model1.score(x_test, y_test)  
result2 = model2.score(x_test, y_test)  
result3 = model3.score(x_test, y_test)  
result4 = model4.score(x_test, y_test)  
result5 = model5.score(x_test, y_test)  
result6 = model6.score(x_test, y_test)  
result7 = model7.score(x_test, y_test)  

from sklearn.metrics import accuracy_score, r2_score
y_predict1 = model1.predict(x_test)
# y_predict2 = model2.predict(x_test)
# y_predict3 = model3.predict(x_test)
# y_predict4 = model4.predict(x_test)
# y_predict5 = model5.predict(x_test)
# y_predict6 = model6.predict(x_test)
# y_predict7 = model7.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict1)

print("Perceptron : ", result1)
print("LinearSVC : ", result2)
print("SVC : ", result3)
print("KNeighborsClassifier : ", result4)
print("LogisticRegression : ", result5)
print("DecisionTreeClassifier : ", result6)
print("RandomForestClassifier : ", result7)
# print("accuracy_score : ", acc)
print("r2_score : ", r2)

# Perceptron :  0.02247191011235955
# LinearSVC :  0.0
# SVC :  0.0
# KNeighborsClassifier :  0.0
# LogisticRegression :  0.0
# DecisionTreeClassifier :  0.0
# RandomForestClassifier :  0.0