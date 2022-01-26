import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA       # PCA : 고차원의 데이터를 저차원의 데이터로 환원, 선을 하나 그어서 거기에 맞춤, 컬럼을 압축(차원 축소)
                                            # fit만 가능

# model = SVC()
# model = make_pipeline(MinMaxScaler(), PCA(), SVC())
model = Pipeline([("mm", MinMaxScaler()),("svc",SVC())])

#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)  

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ", result)


# model.score :  1.0
