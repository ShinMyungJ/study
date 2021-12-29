# 모델링 구성

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
import numpy as np

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

augment_size = 40000
randidx =  np.random.randint(x_train.shape[0], size = augment_size)   # 랜덤한 정수값을 생성   / x_train.shape[0] = 60000이라고 써도 된다.

print(randidx.shape)       # (40000,)
print(type(randidx))       # <class 'numpy.ndarray'>
print(x_train.shape[0])    # 60000 
print(x_train.shape[1])    # 28
print(x_train.shape[2])    # 28
print(np.min(randidx),np.max(randidx))  # 1 59999

x_agumented = x_train[randidx].copy()
y_agumented = y_train[randidx].copy()

print(x_agumented.shape)   # (40000, 28, 28)
print(y_agumented.shape)   # (40000,)

x_agumented = x_agumented.reshape(x_agumented.shape[0], 
                                  x_agumented.shape[1],
                                  x_agumented.shape[2],1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_agumented = train_datagen.flow(x_agumented, y_agumented,
                                 batch_size=augment_size, shuffle=False,
                                 ).next()[0]


x_train = np.concatenate((x_train, x_agumented))       # concatenate 괄호 두 개인 이유
y_train = np.concatenate((y_train, y_agumented))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) # (60000, 10)

model = Sequential() 
model.add(Conv2D(7, kernel_size = (3,3), input_shape = (28,28,1))) 
model.add(Conv2D(5, (3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu'))         
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto',
                   verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함
mcp = ModelCheckpoint (monitor = 'val_acc', mode = 'min', verbose = 1, save_best_only=True,
                       filepath = './_ModelCheckPoint/keras27_5_MCP.hdf5')
model.fit(x_train, y_train, epochs=3, batch_size=64,
          validation_split=0.1111, callbacks=[es,mcp])



model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)


print(y_test)
print(y_test.shape, y_predict.shape)     # (10000, 10) (10000, 10)

y_predict=np.argmax(y_predict, axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score   # accuracy_score 분류에서 사용
a = accuracy_score(y_test, y_predict)
print('acc score:', a)
