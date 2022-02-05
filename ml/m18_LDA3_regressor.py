# 회귀 데이터들 몽땅 집어넣고 LDA2와 동일하게 만드시오
# 보스턴, 디아벳, 캘리포니아!!

import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
# datasets = load_boston()
# datasets = load_diabetes()
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

# print(np.unique(y))
print("LDA 전 : ", x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True   # default : none
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y = y * 1000
y_train = np.round(y_train)

# 소숫점 자리 구하기
##############################################################################################
b = []
for i in y:
    b.append(len(str(i).split('.')[1]))
    
print(np.unique(b, return_counts=True))
###############################################################################################

# pca = PCA(n_components=25)
lda = LinearDiscriminantAnalysis(n_components=5)          # (최대 피쳐 수 or y의 라벨의 수) - 1 보다 크게 n_component를 넣을 수 없음. 여기는 y값이 2개라서 1만 가능.
# x = pca.fit_transform(x)
# x_train = lda.fit_transform(x_train, y_train)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA 후 : ", x_train.shape)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
# model = XGBClassifier()
model = XGBRegressor()


#3. 훈련
# model.fit(x_train, y_train, eval_metric='error')
model.fit(x_train, y_train)
# model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# diabetes
# LDA 전 :  (442, 10)
# LDA 후 :  (353, 5)
# 결과 :  0.34680264756770585

# boston
# LDA 전 :  (506, 13)
# LDA 후 :  (404, 5)
# 결과 :  0.9016514433565562

# fetch_california_housing
# LDA 전 :  (20640, 8)
# LDA 후 :  (16512, 5)
# 결과 :  0.6968932596722985

