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

print(randidx.shape)       # (40000,)
print(type(randidx))       # <class 'numpy.ndarray'>
print(x_train.shape[0])    # 60000 
print(x_train.shape[1])    # 28
print(x_train.shape[2])    # 28
print(np.min(randidx),np.max(randidx))  # 1 59999

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)   # (40000, 28, 28)
print(y_augmented.shape)   # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],1)

x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size, shuffle=False,
                                 ).next()[0]

print(x_augmented)              
print(x_augmented.shape)        # (40000, 28, 28, 1)


x_train = np.concatenate((x_train, x_augmented))       # concatenate 괄호 두 개인 이유
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)    # (100000, 28, 28) == x_train + x_augmented // (100000,) == y_train + y_augmented


# print(x_train[0].reshape(28*28).shape)    # (784,)
# print(x_train[0].shape)     # (28, 28)
# print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape)  # (100, 28, 28, 1)

# augment_size = 100      # augment : 증강 / 증폭하는 크기가 100개
# x_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),     # x_train의 0번째를 확대하겠다. / x가 된다.
#     np.zeros(augment_size),                                                   # y 가 된다.
#     batch_size=augment_size,
#     shuffle = False
#     ).next()

# print(type(x_data))     # <class 'tuple'>   // IDG
# print(x_data)
# print(x_data[0].shape, x_data[1].shape)     # (100, 28, 28, 1) (100,)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7,7, i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap = 'gray')
# plt.show()