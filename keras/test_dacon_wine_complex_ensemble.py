from sklearn.datasets import load_wine
from tensorflow.keras.models import load_model
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

#1 데이터
path = "./_data/wine/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 

submission = pd.read_csv(path+"sample_Submission.csv") #제출할 값
y = train['quality']
x = train.drop(['id', 'quality'], axis =1) # , 'pH', 'free sulfur dioxide', 'residual sugar'
print(x.shape)
# x = train #.drop(['casual','registered','count'], axis =1) #

le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
label = x['type']
le.fit(label)
x['type'] = le.transform(label)

print(x)                          # type column의 white, red를 0,1로 변환
print(x.shape)                    # (3231, 12)

from tensorflow.keras.utils import to_categorical

test_file = test_file.drop(['id'], axis=1) # , 'pH', 'free sulfur dioxide', 'residual sugar'
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 49)

rf = RandomForestClassifier(n_estimators=200, random_state=66)
lr = LogisticRegression()
tree = DecisionTreeClassifier(random_state=59)
lgbm = LGBMClassifier()
svm_clf = SVC()

rf.fit(x_train, y_train)
lr.fit(x_train, y_train)
tree.fit(x_train, y_train)
svm_clf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)
lr_pred = lr.predict(x_test)
tree_pred = tree.predict(x_test)
svm_pred = svm_clf.predict(x_test)

new_data = np.array([rf_pred, lr_pred, tree_pred, svm_pred])
# print(new_data.shape)
new_data = np.transpose(new_data)
print(new_data.shape)

lgbm.fit(new_data, y_test)
lgbm_pred = lgbm.predict(new_data)
print("정확도 : {0:.4f}".format(accuracy_score(y_test, lgbm_pred)))
print(lgbm_pred.shape)
print(new_data.shape)

rf_pred2 = rf.predict(test_file)
lr_pred2 = lr.predict(test_file)
tree_pred2 = tree.predict(test_file)
svm_pred2 = svm_clf.predict(test_file)

test_data = np.array([rf_pred2, lr_pred2, tree_pred2, svm_pred2])
test_data = np.transpose(test_data)


############################### 제출용 ########################################
y_pred = lgbm.predict(test_data)
print(y_pred[:5])
submission['quality'] = y_pred
submission.to_csv(path + "tree31.csv", index = False)