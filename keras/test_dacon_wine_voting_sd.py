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
# one_hot = to_categorical(y,num_classes=len(np.unique(y)))

test_file = test_file.drop(['id'], axis=1) # , 'pH', 'free sulfur dioxide', 'residual sugar'
label2 = test_file['type']
le.fit(label2)
test_file['type'] = le.transform(label2)

y = train['quality']
# print(y.unique())                # [6 7 5 8 4]
# y = get_dummies(y)
# print(y)                         #        4  5  6  7  8
                                   #  0     0  0  1  0  0
                                   #  1     0  0  0  1  0
                                   #  2     0  0  1  0  0
                                   #  3     0  1  0  0  0
                                   #  4     0  0  0  1  0

# y = to_categorical(y) #<=============== class 개수대로 자동으로 분류 해 준다!!! /// 간단!!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 66)

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
    model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(130, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(80))
    # model.add(BatchNormalization())
    model.add(Dense(50))
    # model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))


#3. 컴파일, 훈련
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = mlp_model()

# 서로 다른 모델을 만들어 합친다
model1 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model1._estimator_type="classifier" 
model2 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model2._estimator_type="classifier"
model3 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model3._estimator_type="classifier"
model4 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model4._estimator_type="classifier"
model5 = KerasClassifier(build_fn = mlp_model, epochs = 200, verbose = 1)
model5._estimator_type="classifier"

ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3), ('model4', model4), ('model5', model5)]
                                , voting = 'soft')
ensemble_clf.fit(x_train, y_train)


################################ 제출용 ########################################

y_pred = ensemble_clf.predict(test_file)
submission['quality'] = y_pred
submission.to_csv(path + "em20.csv", index = False)