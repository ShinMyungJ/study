from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

dataset = load_boston()

x = dataset.data
y = dataset.target

import pandas as pd
xx = pd.DataFrame(x, columns=dataset.feature_names)

#상관분석  (공분석)
# Y를 넣고 상관관계 분서
'''
xx['price'] = y
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(data= xx.corr(), square=True, annot=True, cbar=True)
plt.show()
#---12개로 줄이면 506,12,1,1 혹은 506,4,3,1
#줄여서 작업해라
'''
x = xx.drop(['CHAS'],axis=1) #---> Df
x = x.to_numpy()
# print(x)

from sklearn.metrics import r2_score

#데이터

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
         train_size = 0.7, shuffle = True, random_state = 66) 

scaler = StandardScaler()

n = x_train.shape[0]# 
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train) 
print(x_train_transe.shape) #354,13 506,12
x_train = x_train_transe.reshape(n,2,2,3) 

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(m,2,2,3)

print(x_train.shape)

model = Sequential()
model.add(Conv2D(128, kernel_size=(4,4),padding ='same',strides=1, input_shape = (2,2,3)))#
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding ='same', activation='relu'))#<------------
# model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),padding ='same', activation='relu'))
# model.add(MaxPooling2D())
model.add(Flatten())
# print("==========================================★")
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(1))

#model.summary()#3,153
#3. 컴파일, 훈련

opt="adam"
model.compile(loss = 'mse', optimizer = opt) # metrics=['accuracy'] 영향을 미치지 않는다
########################################################################
# model.compile(loss = 'mse', optimizer = 'adam')
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
epoch = 10000
patience_num = 500
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")
filepath = "./_ModelCheckPoint/"
filename = '{epoch:04d}-{val_loss:4f}.hdf5' #filepath + datetime
model_path = "".join([filepath,'k35_cnn_boston_',datetime,"_",filename])
es = EarlyStopping(monitor='val_loss', patience=patience_num, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
hist = model.fit(x_train, y_train, epochs = epoch, validation_split=0.2, callbacks=[es,mcp], batch_size = 50)
end = time.time() - start
print('시간 : ', round(end,2) ,'초')
########################################################################
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 : ",r2)
print("epochs :",epoch)
'''
<<dnn>>
<<ModelCheckpoint 우수>>
Epoch 01423: val_loss did not improve from 6.13479
4/4 [==============================] - 0s 0s/step - loss: 5.5674
loss :  5.567386627197266
R2 :  0.9333908703456402
<<Drop OUT>>
Epoch 00097: val_loss did not improve from 25.87711
4/4 [==============================] - 0s 997us/step - loss: 18.0372
loss :  18.037235260009766
R2 :  0.7841995627355511
<<CNN>>
시간 :  44.89 초
5/5 [==============================] - 0s 801us/step - loss: 9.4039
loss :  9.403949737548828
R2 :  0.8861742911169088
Epoch 00909: val_loss did not improve from 10.08448
시간 :  23.86 초
5/5 [==============================] - 0s 748us/step - loss: 7.7454
loss :  7.745367050170898
R2 :  0.906249825286233
'''