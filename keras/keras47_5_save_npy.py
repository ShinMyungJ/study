import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train',
    target_size=(150, 150),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=200,
    class_mode='binary',
    shuffle=True
)       # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test',
    target_size=(150,150),
    batch_size=200,
    class_mode='binary'    
)       # Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001B7BC3D4F70>

print(xy_train[0][0].shape, xy_train[0][1].shape)       # (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)         # (120, 150, 150, 3) (120,)

np.save('./_save_npy/keras47_4_train_x.npy', arr=xy_train[0][0])
np.save('./_save_npy/keras47_4_train_y.npy', arr=xy_train[0][1])
np.save('./_save_npy/keras47_4_test_x.npy', arr=xy_test[0][0])
np.save('./_save_npy/keras47_4_test_y.npy', arr=xy_test[0][1])
