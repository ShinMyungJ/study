from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest'          
    )                      

augment_size = 100      # augment : 증강 / 증폭하는 크기가 100개

print(x_train[0].reshape(28*28).shape)    # (784,)
print(x_train[0].shape)     # (28, 28)
print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1).shape)  # (100, 28, 28, 1)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),     # x_train의 0번째를 확대하겠다. / x가 된다.
    np.zeros(augment_size),                                                   # y 가 된다.
    batch_size=augment_size,
    shuffle = False
    ).next()

print(type(x_data))     # <class 'tuple'>   // IDG
print(x_data)
print(x_data[0].shape, x_data[1].shape)     # (100, 28, 28, 1) (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap = 'gray')
plt.show()


# np.tile(A, reps) A를 reps 만큼 반복한다.
# reshape(-1,28,28,1) 
# flow는 x와 y가 모두 필요하다.