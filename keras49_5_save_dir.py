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

augment_size = 10
randidx =  np.random.randint(x_train.shape[0], size = augment_size)   # 랜덤한 정수값을 생성   / x_train.shape[0] = 60000이라고 써도 된다.

print(randidx.shape)       # (10,)
print(type(randidx))       # <class 'numpy.ndarray'>
print(x_train.shape[0])    # 60000 
print(x_train.shape[1])    # 28
print(x_train.shape[2])    # 28
print(np.min(randidx),np.max(randidx))  # 1119 58554


x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   # (10, 28, 28)
print(y_augmented.shape)   # (10,)


x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


import time
start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='../_temp/').next()[0]

end_time = time.time() - start_time

print('걸린 시간:', round(end_time,3),'초')