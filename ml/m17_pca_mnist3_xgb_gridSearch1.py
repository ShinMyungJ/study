import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸것

# m17-2결과를 뛰어넘어랐!!!

# parameters = [
#     {"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
#      "max_depth" : [4, 5, 6]},
#     {"n_estimators" : [90, 100, 110], "learning_rate" : [0.1, 0.001, 0.01],
#      "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
#     {"n_estimators" : [90, 110], "learning_rate" : [0.1, 0.001, 0.5],
#      "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1],
#      "colsample_bylevel" : [0.6, 0.7, 0.9]}
# ]
# n_jobs = -1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(len(x_train),-1)
x_test = x_test.reshape(len(x_test),-1)

scaler =MinMaxScaler()   
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)

# 0.95 = n_components=154
# 0.99 : n_components=331
# 0.999 : n_components=486
# 1.0 : n_components=706, 713

pca = PCA(n_components=331)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

#####################################################
# 실습 
# pca를 통해 0.95 이상인 n_component가 몇개???????
# 0.95 : n_components=154
# 0.99 : n_components=331
# 0.999 : n_components=486
# 1.0 : n_components=706, 713
# np.argmax 써라
#####################################################

parameters = [{"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
      "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]}]

#2. 모델구성

model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), parameters, cv=3, verbose=3,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
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



# 1. 나의 최고의 DNN
# time = 508.5025 초
# acc =  0.9490000009536743

# 2. 나의 최고의 CNN
# loss :  0.09230969846248627
# acc :  0.9832000136375427
# time :  932.6304 초

# 3. PCA 0.95
# accurcy :  0.9690999984741211
# 걸린시간 :  926.1452 초

# 4. PCA 0.99
# loss :  0.1548701971769333
# time = 913.6436 초
# acc = 0.9684000015258789

# 5. PCA 0.999
# loss :  0.21999746561050415
# time = 1245.2889 초
# acc = 0.9621000289916992

# 6. PCA 1.0
# loss :  0.2078629434108734
# time = 993.9652 초
# acc = 0.9646999835968018



