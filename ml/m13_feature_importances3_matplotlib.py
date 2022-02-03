import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# model = DecisionTreeClassifier(max_depth=5)
model = RandomForestClassifier(max_depth=5)
# model = GradientBoostingClassifier()
# model = XGBClassifier()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

# print("DecisionTreeClassifier : ", result)
# print("RandomForestClassifier : ", result)
# print("XGBClassifier : ", result)
print("GradientBoostingClassifier : ", result)
print("accuracy_score : ", acc)

print(model.feature_importances_)

# DecisionTreeClassifier :  1.0
# accuracy_score :  1.0
# [0.         0.01695274 0.90455226 0.078495  ]

# RandomForestClassifier :  1.0
# accuracy_score :  1.0
# [0.09829701 0.02314989 0.41301676 0.46553634]

# XGBClassifier :  1.0
# accuracy_score :  1.0
# [0.0110771  0.02904884 0.75245064 0.20742337]

# GradientBoostingClassifier :  1.0
# accuracy_score :  1.0
# [0.00138074 0.01485225 0.60335374 0.38041328]

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
    
plot_feature_importances_dataset(model)
plt.show()