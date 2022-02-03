# 맹그러봐 subplot 이용해서 4개의 모델을 한 화면에 그래프로

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
model2 = RandomForestClassifier(max_depth=5, random_state=42)
model3 = GradientBoostingClassifier(random_state=42)
model4 = XGBClassifier()

#3. 컴파일, 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 평가, 예측

result1 = model1.score(x_test, y_test)  
result2 = model2.score(x_test, y_test)  
result3 = model3.score(x_test, y_test)  
result4 = model4.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
y_predict2 = model2.predict(x_test)
y_predict3 = model3.predict(x_test)
y_predict4 = model4.predict(x_test)
acc1 = accuracy_score(y_test, y_predict1)
acc2 = accuracy_score(y_test, y_predict2)
acc3 = accuracy_score(y_test, y_predict3)
acc4 = accuracy_score(y_test, y_predict4)

print("DecisionTreeClassifier : ", result1)
print("RandomForestClassifier : ", result2)
print("GradientBoostingClassifier : ", result3)
print("XGBClassifier : ", result4)
print("accuracy_score : ", acc1)
print("accuracy_score : ", acc2)
print("accuracy_score : ", acc3)
print("accuracy_score : ", acc4)

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    
plt.subplot(2, 2, 1)
plot_feature_importances_dataset(model1)
plt.subplot(2, 2, 2)
plot_feature_importances_dataset(model2)
plt.subplot(2, 2, 3)
plot_feature_importances_dataset(model3)
plt.subplot(2, 2, 4)
plot_feature_importances_dataset(model4)
    
plt.show()