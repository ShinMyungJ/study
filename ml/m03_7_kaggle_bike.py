import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
# print(submit.shape)     # (6493, 2)
# print(submit_file.columns)
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

print(x.columns)                # column의 갯수가 8개로 줄음
print(x.shape)                  # (10886, 8)

y = train['count']
print(y)                       
print(y.shape)                  # (10886,)

print(submit_file.columns)

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
result1 = model1.score(x_test, y_test)  
result2 = model2.score(x_test, y_test)  
result3 = model3.score(x_test, y_test)  
result4 = model4.score(x_test, y_test)  
result5 = model5.score(x_test, y_test)  
result6 = model6.score(x_test, y_test)  
result7 = model7.score(x_test, y_test)  

# from sklearn.metrics import accuracy_score
# y_predict1 = model1.predict(x_test)
# y_predict2 = model2.predict(x_test)
# y_predict3 = model3.predict(x_test)
# y_predict4 = model4.predict(x_test)
# y_predict5 = model5.predict(x_test)
# y_predict6 = model6.predict(x_test)
# y_predict7 = model7.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

print("Perceptron : ", result1)
print("LinearSVC : ", result2)
print("SVC : ", result3)
print("KNeighborsClassifier : ", result4)
print("LogisticRegression : ", result5)
print("DecisionTreeClassifier : ", result6)
print("RandomForestClassifier : ", result7)
# print("accuracy_score : ", acc)

# Perceptron :  0.9333333333333333
# LinearSVC :  0.9666666666666667
# SVC :  0.9666666666666667
# KNeighborsClassifier :  0.9666666666666667
# LogisticRegression :  1.0
# DecisionTreeClassifier :  0.9666666666666667
# RandomForestClassifier :  0.9666666666666667