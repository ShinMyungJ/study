# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출!

# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_iris
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

y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model_1 = DecisionTreeClassifier(max_depth=5)
model_2 = RandomForestClassifier(max_depth=5)
model_3 = GradientBoostingClassifier(max_depth=5)
model_4 = XGBClassifier(max_depth=5)

#3. 컴파일, 훈련
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
model_4.fit(x_train, y_train)

#4. 평가, 예측

result_1 = model_1.score(x_test, y_test)  
result_2 = model_2.score(x_test, y_test)  
result_3 = model_3.score(x_test, y_test)  
result_4 = model_4.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict_1 = model_1.predict(x_test)
y_predict_2 = model_2.predict(x_test)
y_predict_3 = model_3.predict(x_test)
y_predict_4 = model_4.predict(x_test)
acc_1 = accuracy_score(y_test, y_predict_1)
acc_2 = accuracy_score(y_test, y_predict_2)
acc_3 = accuracy_score(y_test, y_predict_3)
acc_4 = accuracy_score(y_test, y_predict_4)

print("DecisionTreeClassifier : ", result_1)
print("accuracy_score : ", acc_1)
print(model_1.feature_importances_)

print("RandomForestClassifier : ", result_2)
print("accuracy_score : ", acc_2)
print(model_2.feature_importances_)

print("GradientBoostingClassifier : ", result_3)
print("accuracy_score : ", acc_3)
print(model_3.feature_importances_)

print("XGBClassifier : ", result_4)
print("accuracy_score : ", acc_4)
print(model_4.feature_importances_)

# DecisionTreeClassifier :  0.9333333333333333
# accuracy_score :  0.9333333333333333
# [0.0125026  0.         0.53835801 0.44913938]

# RandomForestClassifier :  0.9
# accuracy_score :  0.9
# [0.10013035 0.02902692 0.42459887 0.44624385]

# GradientBoostingClassifier :  0.9666666666666667
# accuracy_score :  0.9666666666666667
# [0.00784771 0.01012783 0.24292126 0.73910321]

# XGBClassifier :  0.9
# accuracy_score :  0.9
# [0.01835513 0.0256969  0.6204526  0.33549538]


# 결과 비교
# 1. DecisionTree
# 기존 acc : 0.9333333333333333
# 컬럼삭제 후 acc : 0.9333333333333333

# 2. RandomForestClassifier
# 기존 acc : 0.9
# 컬럼삭제 후 acc : 0.9666666666666667

# 3. GradientBoostingClassifier
# 기존 acc : 0.9666666666666667
# 컬럼삭제 후 acc : 0.9333333333333333

# 4. XGBClassifier
# 기존 acc : 0.9
# 컬럼삭제 후 acc : 0.9666666666666667