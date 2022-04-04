import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.keras.preprocessing import image
import cv2

from tensorflow.python.keras.layers.core import Dropout

# path = 'D:/_data/men_women/men'
# Load_image = image.load_img(path)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)

# D:\_data\image\brain

train_generator = train_datagen.flow_from_directory(
    'D:/_data/wow',
    target_size=(256,256),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=500,
    # classes=None
    class_mode='binary',
    subset='training',
    shuffle=True,
)       # Found 2317 images belonging to 2 classes.

validation_generator = train_datagen.flow_from_directory(
    'D:/_data/wow',
    target_size=(256,256),
    batch_size=500,
    # classes=None
    class_mode='binary',
    subset='validation'    
)       # Found 992 images belonging to 2 classes.

# np.save('D:/_save_npy/keras48_4_train_x.npy', arr=train_generator[0][0])
# np.save('D:/_save_npy/keras48_4_train_y.npy', arr=train_generator[0][1])
# np.save('D:/_save_npy/keras48_4_test_x.npy', arr=validation_generator[0][0])
# np.save('D:/_save_npy/keras48_4_test_y.npy', arr=validation_generator[0][1])

# x_train = np.load('D:/_save_npy/keras48_4_train_x.npy')
# y_train = np.load('D:/_save_npy/keras48_4_train_y.npy')
# x_test = np.load('D:/_save_npy/keras48_4_test_x.npy')
# y_test = np.load('D:/_save_npy/keras48_4_test_y.npy')

x_train_noised = train_generator[0][0] + np.random.normal(0, 0.05, size=train_generator[0][0].shape)
x_test_noised = validation_generator[0][0] + np.random.normal(0, 0.05, size=validation_generator[0][0].shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, input_shape=(256, 256, 3), kernel_size=3, padding='same', strides=1, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(3, 3, padding='same'))
    
    return model
    
model = autoencoder(hidden_layer_size=154)      # pca 95% -> 154

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, train_generator[0][0], epochs=100)

output = (model.predict(x_test_noised)*255).astype(np.uint8)
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15),
      (ax6, ax7, ax8, ax9, ax10)) = \
          plt.subplots(3, 5, figsize=(20, 7))
          
# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(validation_generator[0][0][random_images[i]])
    if i ==0:
        ax.imshow(validation_generator[0][0][random_images[i]])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]])
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]])
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()