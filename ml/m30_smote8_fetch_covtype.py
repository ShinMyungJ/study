from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

print(pd.Series(y_train).value_counts())
smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
                    base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                    gamma=0, gpu_id=0, importance_type=None,
                    interaction_constraints='', learning_rate=0.300000012,
                    max_delta_step=0, max_depth=9, min_child_weight=1,
                    monotone_constraints='()', n_estimators=5000,
                    num_parallel_tree=1, objective='multi:softprob',
                    predictor='gpu_predictor', random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=None, subsample=1,
                    tree_method='gpu_hist', validate_parameters=1, verbosity=None
                      )

import joblib
# path = "D:/_save_npy/"
joblib.dump(model, "m30_model.dat", compress = 1)

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

score = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('라벨 : ', np.unique(y, return_counts=True))

print("걸린시간 : ", round(end, 4), "초")
print("model.score : ", score)
print("accuracy score : ", round(accuracy_score(y_test, y_predict),4))
print("f1 score : ", round(f1_score(y_test, y_predict, average='macro'),4))

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
