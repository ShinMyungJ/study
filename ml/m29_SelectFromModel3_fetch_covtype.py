
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
from sklearn.metrics import acc_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
#print(dataset.feature_names)
''' 
['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 
'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 
'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 
'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 
'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38',
'Soil_Type_39']
'''
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
                     gpu_id=0), parameters, cv=kfold, verbose=1,    # model : 어떤 모델도 다 쓸 수 있다, parameters : 해당 모델의 파라미터 사용
                     refit=True)

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='mlogloss',              # rmse, mae, logloss, error
          early_stopping_rounds=10,        # mlogloss, merror 
          )     

end = time.time() - start


#4. 평가, 예측
from sklearn.metrics import accuracy_score

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
#               max_depth=5, min_child_weight=1, missing=nan,
#               monotone_constraints='()', n_estimators=200, n_jobs=12,
#               num_parallel_tree=1, objective='multi:softprob',
#               predictor='gpu_predictor', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.3, 'colsample_bytree': 1}
# best_score_ :  0.8785113918514471
# model.score :  0.8806829427811674
# accuracy_score :  0.8806829427811674
# 최적 튠 ACC :  0.8806829427811674
# 걸린시간 :  399.4673 초
# [0.09355187 0.00874314 0.00467361 0.01441145 0.00712814 0.01328278
#  0.00870108 0.01063894 0.00595461 0.01260113 0.06764754 0.03063705
#  0.03409881 0.02612644 0.00401617 0.04530528 0.01868684 0.04132605
#  0.0065575  0.00605951 0.00142178 0.00716646 0.01185246 0.01147147
#  0.01239016 0.04338493 0.01089295 0.00330044 0.00072532 0.00571225
#  0.01149631 0.00470442 0.00564067 0.01088508 0.01959529 0.05508342
#  0.02838779 0.01683726 0.00680128 0.00507491 0.01585812 0.0032563
#  0.02867791 0.01622727 0.02123862 0.03905323 0.01640995 0.00528165
#  0.01661629 0.00273677 0.00816693 0.0334583  0.04861542 0.01143071]
# [0.00072532 0.00142178 0.00273677 0.0032563  0.00330044 0.00401617
#  0.00467361 0.00470442 0.00507491 0.00528165 0.00564067 0.00571225
#  0.00595461 0.00605951 0.0065575  0.00680128 0.00712814 0.00716646
#  0.00816693 0.00870108 0.00874314 0.01063894 0.01088508 0.01089295
#  0.01143071 0.01147147 0.01149631 0.01185246 0.01239016 0.01260113
#  0.01328278 0.01441145 0.01585812 0.01622727 0.01640995 0.01661629
#  0.01683726 0.01868684 0.01959529 0.02123862 0.02612644 0.02838779
#  0.02867791 0.03063705 0.0334583  0.03409881 0.03905323 0.04132605
#  0.04338493 0.04530528 0.04861542 0.05508342 0.06764754 0.09355187]
# =============================================
# (464809, 54) (116203, 54)

acc_list = []
th_list = []

for thresh in thresholds:    
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit= True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    # print(select_x_train.shape,select_x_test.shape)
    
    selection_model = XGBClassifier(tree_method = 'gpu_hist',
                     predictor = 'gpu_predictor',
                     gpu_id=0)
    #3 훈련
    selection_model.fit(select_x_train, y_train, eval_metric='merror')

    #4 평가 예측
    score = selection_model.score(select_x_test, y_test)
    select_y_pred = selection_model.predict(select_x_test)

    select_acc = accuracy_score(y_test, select_y_pred)
    # print("select_Score : ", score)
    # print("select_acc : ", select_acc)    
    print("Thresh = %.3f, n=%d, acc :%2f%%" %(thresh,select_x_train.shape[1],select_acc*100))
    acc_list.append(select_acc)
    th_list.append(thresh)
    

index_max_acc = acc_list.index(max(acc_list))
print(index_max_acc)
drop_list = np.where(model.best_estimator_.feature_importances_ < th_list[index_max_acc])
print(drop_list)
x,y = fetch_covtype(return_X_y=True)

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


# 피처 줄인다음에!!!
# 다시 모델해서 결과 비교

# (464809, 54) (116203, 54)
# Thresh=0.001, n=54, acc: 86.94%
# (464809, 53) (116203, 53)
# Thresh=0.002, n=53, acc: 87.14%
# (464809, 52) (116203, 52)
# Thresh=0.003, n=52, acc: 86.84%
# (464809, 51) (116203, 51)
# Thresh=0.004, n=51, acc: 87.25%
# (464809, 50) (116203, 50)
# Thresh=0.004, n=50, acc: 86.92%
# (464809, 49) (116203, 49)
# Thresh=0.004, n=49, acc: 86.82%
# (464809, 48) (116203, 48)
# Thresh=0.005, n=48, acc: 87.24%
# (464809, 47) (116203, 47)
# Thresh=0.005, n=47, acc: 87.19%
# (464809, 46) (116203, 46)
# Thresh=0.006, n=46, acc: 87.23%
# (464809, 45) (116203, 45)
# Thresh=0.006, n=45, acc: 87.26%
# (464809, 44) (116203, 44)
# Thresh=0.006, n=44, acc: 86.98%
# (464809, 43) (116203, 43)
# Thresh=0.006, n=43, acc: 86.97%
# (464809, 42) (116203, 42)
# Thresh=0.007, n=42, acc: 87.05%
# (464809, 41) (116203, 41)
# Thresh=0.007, n=41, acc: 87.18%
# (464809, 40) (116203, 40)
# Thresh=0.007, n=40, acc: 87.35%
# (464809, 39) (116203, 39)
# Thresh=0.007, n=39, acc: 86.73%
# (464809, 38) (116203, 38)
# Thresh=0.008, n=38, acc: 86.08%
# (464809, 37) (116203, 37)
# Thresh=0.009, n=37, acc: 86.33%
# (464809, 36) (116203, 36)
# Thresh=0.009, n=36, acc: 86.08%
# (464809, 35) (116203, 35)
# Thresh=0.010, n=35, acc: 85.29%
# (464809, 34) (116203, 34)
# Thresh=0.011, n=34, acc: 85.45%
# (464809, 33) (116203, 33)
# Thresh=0.012, n=33, acc: 83.69%
# (464809, 32) (116203, 32)
# Thresh=0.012, n=32, acc: 78.77%
# (464809, 31) (116203, 31)
# Thresh=0.012, n=31, acc: 78.73%
# (464809, 30) (116203, 30)
# Thresh=0.012, n=30, acc: 78.72%
# (464809, 29) (116203, 29)
# Thresh=0.012, n=29, acc: 78.31%
# (464809, 28) (116203, 28)
# Thresh=0.013, n=28, acc: 72.81%
# (464809, 27) (116203, 27)
# Thresh=0.013, n=27, acc: 72.85%
# (464809, 26) (116203, 26)
# Thresh=0.013, n=26, acc: 72.85%
# (464809, 25) (116203, 25)
# Thresh=0.014, n=25, acc: 72.84%
# (464809, 24) (116203, 24)
# Thresh=0.014, n=24, acc: 72.67%
# (464809, 23) (116203, 23)
# Thresh=0.014, n=23, acc: 72.67%
# (464809, 22) (116203, 22)
# Thresh=0.018, n=22, acc: 72.59%
# (464809, 21) (116203, 21)
# Thresh=0.018, n=21, acc: 72.44%
# (464809, 20) (116203, 20)
# Thresh=0.019, n=20, acc: 72.33%
# (464809, 19) (116203, 19)
# Thresh=0.021, n=19, acc: 72.10%
# (464809, 18) (116203, 18)
# Thresh=0.023, n=18, acc: 71.93%
# (464809, 17) (116203, 17)
# Thresh=0.023, n=17, acc: 71.83%
# (464809, 16) (116203, 16)
# Thresh=0.024, n=16, acc: 71.25%
# (464809, 15) (116203, 15)
# Thresh=0.024, n=15, acc: 71.22%
# (464809, 14) (116203, 14)
# Thresh=0.024, n=14, acc: 71.21%
# (464809, 13) (116203, 13)
# Thresh=0.025, n=13, acc: 71.14%
# (464809, 12) (116203, 12)
# Thresh=0.026, n=12, acc: 71.13%
# (464809, 11) (116203, 11)
# Thresh=0.028, n=11, acc: 71.07%
# (464809, 10) (116203, 10)
# Thresh=0.030, n=10, acc: 70.97%
# (464809, 9) (116203, 9)
# Thresh=0.032, n=9, acc: 70.79%
# (464809, 8) (116203, 8)
# Thresh=0.039, n=8, acc: 70.33%
# (464809, 7) (116203, 7)
# Thresh=0.040, n=7, acc: 69.71%
# (464809, 6) (116203, 6)
# Thresh=0.043, n=6, acc: 69.71%
# (464809, 5) (116203, 5)
# Thresh=0.043, n=5, acc: 69.11%
# (464809, 4) (116203, 4)
# Thresh=0.044, n=4, acc: 68.75%
# (464809, 3) (116203, 3)
# Thresh=0.053, n=3, acc: 68.46%
# (464809, 2) (116203, 2)
# Thresh=0.062, n=2, acc: 67.88%
# (464809, 1) (116203, 1)
# Thresh=0.072, n=1, acc: 67.37%

# =====================================================================

# Thresh = 0.001, n=54, acc :87.064017%
# Thresh = 0.001, n=53, acc :87.148352%
# Thresh = 0.003, n=52, acc :87.255923%
# Thresh = 0.003, n=51, acc :87.093276%
# Thresh = 0.003, n=50, acc :87.142329%
# Thresh = 0.003, n=49, acc :86.970216%
# Thresh = 0.003, n=48, acc :87.136305%
# Thresh = 0.003, n=47, acc :87.543351%
# Thresh = 0.005, n=46, acc :87.681041%
# Thresh = 0.005, n=45, acc :87.390171%
# Thresh = 0.005, n=44, acc :87.452131%
# Thresh = 0.006, n=43, acc :87.310999%
# Thresh = 0.006, n=42, acc :87.407382%
# Thresh = 0.007, n=41, acc :87.372959%
# Thresh = 0.007, n=40, acc :87.551096%
# Thresh = 0.007, n=39, acc :87.279158%
# Thresh = 0.008, n=38, acc :87.644037%
# Thresh = 0.008, n=37, acc :87.539909%
# Thresh = 0.008, n=36, acc :87.294648%
# Thresh = 0.008, n=35, acc :87.063157%
# Thresh = 0.009, n=34, acc :87.212034%
# Thresh = 0.009, n=33, acc :86.952144%
# Thresh = 0.010, n=32, acc :86.891905%
# Thresh = 0.010, n=31, acc :86.955586%
# Thresh = 0.012, n=30, acc :86.249925%
# Thresh = 0.013, n=29, acc :85.854066%
# Thresh = 0.013, n=28, acc :82.058122%
# Thresh = 0.013, n=27, acc :82.319734%
# Thresh = 0.014, n=26, acc :80.780186%
# Thresh = 0.014, n=25, acc :74.834557%
# Thresh = 0.015, n=24, acc :74.869840%
# Thresh = 0.015, n=23, acc :74.625440%
# Thresh = 0.015, n=22, acc :74.565201%
# Thresh = 0.015, n=21, acc :74.535941%
# Thresh = 0.016, n=20, acc :74.143525%
# Thresh = 0.016, n=19, acc :74.088449%
# Thresh = 0.016, n=18, acc :71.825168%
# Thresh = 0.019, n=17, acc :71.800212%
# Thresh = 0.020, n=16, acc :71.733948%
# Thresh = 0.020, n=15, acc :71.661661%
# Thresh = 0.021, n=14, acc :71.631541%
# Thresh = 0.029, n=13, acc :71.594537%
# Thresh = 0.031, n=12, acc :71.551509%
# Thresh = 0.032, n=11, acc :71.537740%
# Thresh = 0.033, n=10, acc :71.527413%
# Thresh = 0.033, n=9, acc :70.912111%
# Thresh = 0.035, n=8, acc :70.276155%
# Thresh = 0.038, n=7, acc :69.728837%
# Thresh = 0.038, n=6, acc :69.679785%
# Thresh = 0.040, n=5, acc :69.044689%
# Thresh = 0.057, n=4, acc :68.690137%
# Thresh = 0.060, n=3, acc :68.099791%
# Thresh = 0.067, n=2, acc :67.830435%
# Thresh = 0.104, n=1, acc :67.236646%
# 8
# (array([14, 20, 27, 28, 31, 39, 41, 49], dtype=int64),)
# ================================= 수정 후
# Score :  0.8719826510503171
# acc :  0.8719826510503171
# =================================