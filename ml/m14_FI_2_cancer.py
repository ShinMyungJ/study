# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출!

# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x = np.delete(x, [2, 16, 17, 24, 25], axis=1)

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


# DecisionTreeClassifier :  0.9385964912280702
# accuracy_score :  0.9385964912280702
# [0.         0.06054151 0.         0.         0.01193499 0.
#  0.         0.02005078 0.         0.         0.         0.
#  0.         0.01257413 0.         0.00636533 0.02291518 0.00442037
#  0.         0.         0.         0.01642816 0.         0.72839202
#  0.         0.         0.         0.11637753 0.         0.        ]

# RandomForestClassifier :  0.956140350877193
# accuracy_score :  0.956140350877193
# [0.03608223 0.01986276 0.0274873  0.04870807 0.00441869 0.01432417
#  0.03490253 0.07130125 0.00381581 0.004963   0.00739296 0.00322693
#  0.01562481 0.06877015 0.0024987  0.00337377 0.00470459 0.00568534
#  0.00403052 0.00523829 0.13556261 0.01428868 0.1556547  0.10419826
#  0.01131582 0.01649071 0.04560054 0.11650475 0.00787284 0.00609921]
# [4, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 28, 29]

# GradientBoostingClassifier :  0.956140350877193
# accuracy_score :  0.956140350877193
# [1.44025945e-04 5.80192342e-02 1.91292461e-03 1.54234936e-04
#  3.57219818e-03 1.47929412e-03 1.46003812e-03 1.42001711e-02
#  1.63088388e-04 8.66218258e-03 4.21287858e-04 5.50195265e-05
#  1.74322852e-04 1.24428920e-02 9.25379110e-04 1.07987312e-02
#  2.69973311e-03 8.11872271e-04 3.92731897e-03 2.11927674e-03
#  4.66726373e-01 2.83520771e-02 1.08133107e-04 2.53990545e-01
#  3.51303121e-03 5.61556649e-04 2.63025863e-03 1.15260666e-01
#  8.31915175e-04 3.88221839e-03]
# [0, 3, 8, 11, 12, 22]

# XGBClassifier :  0.9736842105263158
# accuracy_score :  0.9736842105263158
# [0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
#  0.0054994  0.09745206 0.00340272 0.00369179 0.00769183 0.00281184
#  0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
#  0.00639412 0.0050556  0.01813928 0.02285904 0.22248559 0.2849308
#  0.00233393 0.         0.00903706 0.11586287 0.00278498 0.00775311]
# [2, 16, 17, 24, 25]


# 결과 비교
# 1. DecisionTree
# 기존 acc : 0.9385964912280702
# 컬럼삭제 후 acc : 0.9298245614035088

# 2. RandomForestClassifier
# 기존 acc : 0.956140350877193
# 컬럼삭제 후 acc : 0.9649122807017544

# 3. GradientBoostingClassifier
# 기존 acc : 0.956140350877193
# 컬럼삭제 후 acc : 0.9736842105263158

# 4. XGBClassifier
# 기존 acc : 0.9736842105263158
# 컬럼삭제 후 acc : 0.9736842105263158