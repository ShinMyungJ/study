from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255       #astype(형 변환), /255(스케일링)
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'liner', 'sigmoid', 'selu', 'elu']
    return {"batch_size" : batchs, "optimizer" : optimizers,
            "drop" : dropout, 'activation' : activation}
    
hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)         # 케라스 형태를 사이킷런 형태로 래핑해줌

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = GridSearchCV(keras_model, hyperparameters, cv=3)

import time
start = time.time()
model.fit(x_train, y_train, verbose=1, epochs=1, validation_split=0.2)
end = time.time() - start

import joblib
# joblib.dump(model.best_estimator_, 'keras55_1_best_estimateor.h5', compress = 1)
joblib.dump(model.best_params_, 'best_keras55_1_best_estimateor11.pkl', compress = 1) # Only best parameters

print("걸린시간 : ", round(end, 3), "초")
print("model.best_params_ : ", model.best_params_)
print("model.best_estimator_ : ", model.best_estimator_)
print("model.best_score_ : ", model.best_score_)
print("model.score : ", model.score)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

#가중치 save