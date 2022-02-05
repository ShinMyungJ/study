import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
# datasets = load_boston()
datasets = fetch_covtype()
# datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)  # (506, 13) -> (20640, 8)
print(np.unique(y))     # [1 2 3 4 5 6 7]

# pca = PCA(n_components=25)
lda = LinearDiscriminantAnalysis()          # (최대 피쳐 수 or y의 라벨의 수) - 1 보다 크게 n_component를 넣을 수 없음. 여기는 y값이 2개라서 1만 가능.
# x = pca.fit_transform(x)
# x = lda.fit_transform(x, y)
x = lda.fit_transform(x, y)

lda.fit(x, y)
x = lda.transform(x)


# print(x)
print(x.shape)          # (506, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric='merror')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# LDA
# (569, 1)
# 결과 :  0.9824561403508771

# PCA(xgboost)
# (569, 30)
# 결과 :  0.9736842105263158

# (569, 25)
# 결과 :  0.9649122807017544

# (506, 11)
# 결과 :  0.891970250038245

# (506, 12)
# 결과 :  0.8874143051779056

# (506, 13)
# 결과 :  0.8996428829583528
