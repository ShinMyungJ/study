import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.datasets import fetch_covtype

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')



#1. 데이터
# datasets = load_iris()
# datasets = load_breast_cancer()
# datasets = load_wine()
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(np.unique(y))
print("LDA 전 : ", x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y  # default : none
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# pca = PCA(n_components=25)
lda = LinearDiscriminantAnalysis()          # (최대 피쳐 수 or y의 라벨의 수) - 1 보다 크게 n_component를 넣을 수 없음. 여기는 y값이 2개라서 1만 가능.
# x = pca.fit_transform(x)
# x_train = lda.fit_transform(x_train, y_train)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA 후 : ", x_train.shape)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
# model.fit(x_train, y_train, eval_metric='error')
model.fit(x_train, y_train, eval_metric='merror')
# model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# iris
# [0 1 2]
# LDA 전 :  (150, 4)
# LDA 후 :  (120, 2)
# 결과 :  1.0

# cancer
# [0 1]
# LDA 전 :  (569, 30)
# LDA 후 :  (455, 1)
# 결과 :  0.9473684210526315

# wine
# [0 1 2]
# LDA 전 :  (178, 13)
# LDA 후 :  (142, 2)
# 결과 :  1.0

# fetch covtype
# [1 2 3 4 5 6 7]
# LDA 전 :  (581012, 54)
# LDA 후 :  (464809, 6)
# 결과 :  0.7878109859470065