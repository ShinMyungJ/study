'''
차원 축소
차원(dimention):
    예를 들어 (1000, 500)의 데이터를 모델링하여 실행 했을 때
    y=w1x1+w2x2....w500x500 + b의 연산을 하게된다. >> 자원 낭비가 심하다. 
    
  
컬럼, 피쳐, 열 
왜 축소를 할까?
'''
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
# datasets = load_boston()
datasets = load_breast_cancer()
# datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)          # (506, 13) -> (20640, 8)

pca = PCA(n_components=25)
x = pca.fit_transform(x)
print(x)
print(x.shape)          # (506, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, eval_metric='error')

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

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
