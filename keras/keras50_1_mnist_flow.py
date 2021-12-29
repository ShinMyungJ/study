# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= False,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    # rotation_range= 5,
    zoom_range = 0.1,              
    # shear_range=0.7,
    fill_mode = 'nearest'          
    )

test_datagen = ImageDataGenerator(
    rescale=1./255
)                      

augment_size = 40000
randidx =  np.random.randint(x_train.shape[0], size = augment_size)   # 랜덤한 정수값을 생성   / x_train.shape[0] = 60000이라고 써도 된다.

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape)   # (40000, 28, 28)
# print(y_augmented.shape)   # (40000,)

minmax_scaler = MinMaxScaler()
x_train_scale = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_train = minmax_scaler.fit_transform(x_train_scale).reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

x_augmented = train_datagen.flow(x_augmented,
                                 y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                #  save_to_dir = '../_temp/'
                                 ).next()[0]

x_test = test_datagen.flow(x_test, y_test,
                           batch_size=augment_size,
                           shuffle=False,
                           ).next()[0]

x_train = np.concatenate((x_train, x_augmented))        # concatenate 괄호 두 개인 이유
y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape, y_train.shape)                     # (100000, 28, 28, 1) (100000,)
# print(x_test.shape, y_test.shape)                       # (10000, 28, 28, 1) (10000,)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분
# print(datetime)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 100(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k50_1_', datetime, '_', filename])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True, filepath=model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
scores = model.evaluate(x_test, y_test)
print("%s: %.2f" %(model.metrics_names[0], scores[0]))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# y_predict = model.predict(x_test)
# print(predict[0])

# 걸린시간 :  132.417 초
# ====================== 1. 기본출력 ========================
# 313/313 [==============================] - 1s 3ms/step - loss: 0.5235 - acc: 0.8187
# loss: 0.52
# acc: 81.87%