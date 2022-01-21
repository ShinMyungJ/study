# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출!

# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

x = np.delete(x, [0, 5, 6], axis=1)

# x= pd.DataFrame(x)
# x = x.drop([0],axis=1)

# print(x.shape) # (150, 3)

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

# DecisionTreeClassifier :  0.9444444444444444
# accuracy_score :  0.9444444444444444
# [0.         0.0555874  0.04078249 0.         0.18739896 0.
#  0.         0.0177651  0.         0.33215293 0.36631311]
# [0, 3, 5, 6, 8]

# RandomForestClassifier :  1.0
# accuracy_score :  1.0
# [0.0211987  0.04610854 0.03859672 0.06939801 0.1524042  0.009869
#  0.02073522 0.17343534 0.10179448 0.14395063 0.22250916]
# [0, 2, 5, 6]

# GradientBoostingClassifier :  0.9166666666666666
# accuracy_score :  0.9166666666666666
# [2.03724529e-02 1.49050410e-02 9.91559080e-04 4.28915418e-04
#  1.13786656e-01 9.13695428e-18 1.35465640e-17 2.54261179e-01
#  5.02357424e-02 2.55016335e-01 2.90002120e-01]
# [2, 3, 5, 6]

# XGBClassifier :  1.0
# accuracy_score :  1.0
# [0.00876542 0.02820693 0.04431904 0.03134441 0.1203698  0.00302756
#  0.00902437 0.14012972 0.01874881 0.44056705 0.15549687]
# [0, 5, 6]


# 결과 비교
# 1. DecisionTree
# 기존 acc : 0.9444444444444444
# 컬럼삭제 후 acc : 0.9166666666666666

# 2. RandomForestClassifier
# 기존 acc : 1.0
# 컬럼삭제 후 acc : 1.0

# 3. GradientBoostingClassifier
# 기존 acc : 0.9166666666666666
# 컬럼삭제 후 acc : 0.9444444444444444

# 4. XGBClassifier
# 기존 acc : 1.0
# 컬럼삭제 후 acc : 1.0