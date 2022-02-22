from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# [실습] 아래꺼를 수정!!!

#1 데이터

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
n = x_train.shape[0]                           
x_train_reshape = x_train.reshape(n,-1)                
x_train_transe = scaler.fit_transform(x_train_reshape) 
x_train = x_train_transe.reshape(x_train.shape)    

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)


#2 모델 구성

input = Input(shape = (32, 32, 3))
vgg16 = VGG16(include_top=False)(input)
hidden1 = GlobalAveragePooling2D()(vgg16)
hidden1 = Dense(512, activation='relu')(hidden1)
hidden1 = Dense(256, activation='relu')(hidden1)
output1 = Dense(100, activation='softmax')(hidden1)

model = Model(inputs=input, outputs=output1)

model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
import time

lr = 1e-5
optimizer = Adam(lr=lr)   # 0.01일때와 0.001, 0.0001일때 성능과 시간 비교

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics='accuracy')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k69_1_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()

hist = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.25, callbacks=[es, reducelr])#, mcp])

end = time.time() - start

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
result = model.predict(x_test)
print("learning rate : ", lr)
print('loss : ', round(loss, 4))
print('accurcy : ', round(acc,4))
print("시간 : ", round(end, 4), '초')

