

# GridSearchCV 적용해서 출력한 값에서
# 피처임포턴스 추출후
# SelectFromModel 맹그러서
# 칼럼 축소후
# 모델구축해서 결과 도출

from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


#1. 데이터 
path = '../_data/winequlity/'   
datasets = pd.read_csv(path+'winequality-white.csv',
                       index_col=None, sep=';', header=0, dtype=float)
print(datasets.shape)   # (4898, 12)
datasets = datasets.values      # numpy로 변환

x = datasets[:, :-1]
y = datasets[:, -1]

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

n_splits = 5        # 3 or 5를 많이씀
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [{"n_estimators" : [100, 200, 300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
      "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]}]

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = PolynomialFeatures()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = RandomizedSearchCV(XGBClassifier(tree_method = 'gpu_hist',
                     predictor = 'gpu_predictor',
                     gpu_id=0,), parameters, cv=kfold, verbose=1,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
                     refit=True)

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='mlogloss',              # rmse, mae, logloss, error
          early_stopping_rounds=10,            # mlogloss, merror 
          )     

end = time.time() - start


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

print("걸린시간 : ", round(end, 4), "초")

print(model.best_estimator_.feature_importances_)
print(np.sort(model.best_estimator_.feature_importances_))
thresholds = np.sort(model.best_estimator_.feature_importances_)

# 최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#               gamma=0, gpu_id=0, importance_type=None,
#               interaction_constraints='', learning_rate=0.3, max_delta_step=0,
#               max_depth=6, min_child_weight=1, missing=nan,
#               monotone_constraints='()', n_estimators=100, n_jobs=12,
#               num_parallel_tree=1, objective='multi:softprob',
#               predictor='gpu_predictor', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None)
# 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.3, 'colsample_bytree': 1}
# best_score_ :  0.622001980868976
# model.score :  0.6785714285714286
# accuracy_score :  0.6785714285714286
# 최적 튠 ACC :  0.6785714285714286
# 걸린시간 :  151.7919 초
# [0.06760225 0.11273944 0.07215688 0.07929648 0.06945042 0.09051217
#  0.07394098 0.07775468 0.07184472 0.0688524  0.21584953]
# [0.06760225 0.0688524  0.06945042 0.07184472 0.07215688 0.07394098
#  0.07775468 0.07929648 0.09051217 0.11273944 0.21584953]

acc_list = []
th_list = []

for thresh in thresholds:    
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit= True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    # print(select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs = -1)
    #3 훈련
    selection_model.fit(select_x_train, y_train, eval_metric='merror')

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_acc = accuracy_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    print("select_acc : ", select_acc)
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
    %(thresh, select_x_train.shape[1], score*100))  
    # print("Thresh = %.3f, n=%d, R2 :%2f%%" %(thresh,select_x_train.shape[1],select_r2*100))
    acc_list.append(select_acc)
    th_list.append(thresh)
    

index_max_acc = acc_list.index(max(acc_list))
print(index_max_acc)
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_acc])
print(drop_list)

x = np.delete(x,drop_list,axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.8, shuffle = True, random_state = 66
         #, stratify= y
         )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = XGBClassifier(n_jobs = -1)
#3 훈련
model.fit(x_train, y_train)

#4 평가 예측
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('================================= 수정 후')
acc = accuracy_score(y_test, y_pred)
print("Score : ", score)
print("acc : ", acc)
print('=================================')

# (3918, 11) (980, 11)
# Thresh=0.073, n=11, ACC: 68.27%
# (3918, 10) (980, 10)
# Thresh=0.074, n=10, ACC: 69.18%
# (3918, 9) (980, 9)
# Thresh=0.074, n=9, ACC: 69.39%
# (3918, 8) (980, 8)
# Thresh=0.075, n=8, ACC: 68.27%
# (3918, 7) (980, 7)
# Thresh=0.076, n=7, ACC: 66.94%
# (3918, 6) (980, 6)
# Thresh=0.077, n=6, ACC: 65.92%
# (3918, 5) (980, 5)
# Thresh=0.085, n=5, ACC: 64.90%
# (3918, 4) (980, 4)
# Thresh=0.092, n=4, ACC: 63.88%
# (3918, 3) (980, 3)
# Thresh=0.101, n=3, ACC: 60.51%
# (3918, 2) (980, 2)
# Thresh=0.103, n=2, ACC: 55.82%

# select_acc :  0.6826530612244898
# select_acc :  0.6877551020408164
# select_acc :  0.6877551020408164
# select_acc :  0.6642857142857143
# select_acc :  0.6693877551020408
# select_acc :  0.6530612244897959
# select_acc :  0.6489795918367347
# select_acc :  0.65
# select_acc :  0.639795918367347
# select_acc :  0.5581632653061225
# select_acc :  0.4887755102040816

# ================================= 수정 후
# Score :  0.689795918367347
# acc :  0.689795918367347
# =================================
# 피처 줄인다음에!!!
# 다시 모델해서 결과 비교

