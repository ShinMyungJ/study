# 13_1번을 가져다가
# 첫번째 칼럼을 제거 후
# 13_1번과 성능 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

x = np.delete(x, [0, 1], axis=1)

# x= pd.DataFrame(x)
# x = x.drop([0],axis=1)

# print(x.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


model = DecisionTreeClassifier(max_depth=5)

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("DecisionTreeClassifier : ", result)
print("accuracy_score : ", acc)

print(model.feature_importances_)

# DecisionTreeClassifier :  1.0
# accuracy_score :  1.0
# [0.01695274 0.90455226 0.078495  ]

# DecisionTreeClassifier :  1.0
# accuracy_score :  1.0
# [0.92015135 0.07984865]