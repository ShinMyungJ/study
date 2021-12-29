from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   # (40000, 28, 28)
print(y_augmented.shape)   # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

xy_train = train_datagen.flow(x_augmented, y_augmented,
                                #  batch_size=32       augment_size, 
                                 shuffle=False)
                                #  ).next()

print(len(xy_train))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit_generator(xy_train, epochs = 3, steps_per_epoch=len(xy_train))


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
acc = accuracy_score(y_test, y_predict)
print('acc score:', acc)
