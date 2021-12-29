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

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size, shuffle=False,
                                 ).next()[0]


# print(x_augmented)
# print(x_augmented.shape)  # (10, 28, 28, 1)
 

# x_train = np.concatenate((x_train, x_augmented))       # concatenate 괄호 두 개인 이유
# y_train = np.concatenate((y_train, y_augmented))
# print(x_train.shape)    # (60010, 28, 28, 1) = x_train + x_augmented


# print(x_train[0][2].shape)   # (28, 1)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,2))
for i in range(59989):
    plt.subplot(2,10, i+1)
    plt.axis('off')
    plt.imshow(x_train[0][i], cmap = 'gray')
plt.show()


'''

1. x_augumented 10개와 x_train 10개를 비교하는 이미지를 출력할 것 
    # sublot(2, 10, ?) 사용
변환 전
변환 후

'''
# print(x_augmented[randidx])