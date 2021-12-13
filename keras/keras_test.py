from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

def f1_score(answer, submission):
    true = answer
    pred = submission
    score = metrics.f1_score(y_true=true, y_pred=pred)
    return score


#1 데이터

path = "./_data/heart_disease/"  
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv")

submission = pd.read_csv(path+"sample_submission.csv") #제출할 값

# print(train)      #features : id  age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target

y = train['target']
x = train.drop(['id', 'target'], axis =1)
test_file = test_file.drop(['id'], axis=1)
# print(type(x))
# print(x.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=x.corr(), square=True, annot=True, cbar=True)
# plt.show()

            #    age       sex        cp  trestbps      chol       fbs   restecg   thalach     exang   oldpeak     slope        ca      thal    target
# target   -0.247806 -0.246289  0.436273 -0.122850  0.036991  0.065821  0.017528  0.365249 -0.396145 -0.421514  0.305994 -0.466289 -0.428530  1.000000

# le = LabelEncoder()                 # 라벨 인코딩은 n개의 범주형 데이터를 0부터 n-1까지 연속적 수치 데이터로 표현
# label = x['type']
# le.fit(label)
# x['type'] = le.transform(label)

# print(x)                        # type column의 텍스트를 숫자로 변환
print(x.shape)                    # (151, 13)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2 모델구성

from keras.layers import BatchNormalization
def mlp_model():
    model = Sequential()
    model.add(Dense(64, input_dim=x.shape[1], activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16))
    # model.add(BatchNormalization())
    model.add(Dense(8))
    # model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model

model = mlp_model()

# 서로 다른 모델을 만들어 합친다
model1 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 1)
model1._estimator_type="classifier" 
model2 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 1)
model2._estimator_type="classifier"
model3 = KerasClassifier(build_fn = mlp_model, epochs = 100, verbose = 1)
model3._estimator_type="classifier"
# model4 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
# model4._estimator_type="classifier"
# model5 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
# model5._estimator_type="classifier"

ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)] #, ('model4', model4), ('model5', model5)
                                , voting = 'soft')

ensemble_clf.fit(x_train, y_train)
y_predict = ensemble_clf.predict(x_test)

print(y_predict.shape, y_test.shape)
print("f1_score : ", f1_score(y_predict, y_test))


################################ 제출용 ########################################

y_pred = ensemble_clf.predict(test_file)
submission['target'] = y_pred
print(submission[:10])
submission.to_csv(path + "heart_voting2.csv", index = False)