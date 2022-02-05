# 맹그러!!!

import numpy as np
from tensorflow.keras.datasets import mnist
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split, GridSearchCV

import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)

print(np.unique(y_train))
# print("LDA 전 : ", x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print("LDA 전 : ", x_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=9)          # (최대 피쳐 수 or y의 라벨의 수) - 1 보다 크게 n_component를 넣을 수 없음. 여기는 y값이 2개라서 1만 가능.
# x = pca.fit_transform(x)
# x_train = lda.fit_transform(x_train, y_train)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print("LDA 후 : ", x_train.shape)

parameters = [{"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
      "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]}]

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = GridSearchCV(XGBClassifier(use_label_encoder=False), parameters, cv=3, verbose=3,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
                     refit=True, n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train, eval_metric='merror')
end = time.time()

#4. 평가, 예측

from sklearn.metrics import accuracy_score, r2_score

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)         # 이렇게하면 best_score_와 model.score 사이에서 실수할 일 없음
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", end - start)

# 최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9,
#               enable_categorical=False, gamma=0, gpu_id=-1,
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=6,
#               n_estimators=300, n_jobs=12, num_parallel_tree=1,
#               objective='multi:softprob', predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', use_label_encoder=False,
#               validate_parameters=1, verbosity=None)
# 최적의 파라미터 :  {'colsample_bytree': 0.9, 'learning_rate': 0.3, 'max_depth': 6, 'n_estimators': 300}
# best_score_ :  0.9174166666666667
# model.score :  0.918
# accuracy_score :  0.918
# 최적 튠 ACC :  0.918
# 걸린시간 :  6337.32866859436

# mnist
# [0 1 2 3 4 5 6 7 8 9]
# LDA 전 :  (60000, 28, 28)
# LDA 후 :  (60000, 9)
# 결과 :  0.9163

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 8)
# 결과 :  0.909

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 7)
# 결과 :  0.8915

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 6)
# 결과 :  0.8676

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 5)
# 결과 :  0.842

# LDA 전 :  (60000, 784)
# LDA 후 :  (60000, 4)
# 결과 :  0.8282