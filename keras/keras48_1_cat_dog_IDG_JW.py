import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

# 1. 데이터

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

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    '../_data/image/cat_dog/training_set',
    target_size = (50,50),                         
    batch_size = 10,
    class_mode = 'binary',
    shuffle = True,
    )           


xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set',
    target_size = (50,50),
    batch_size = 10, 
    class_mode = 'binary',
)

print(xy_train[0][0].shape, xy_train[0][1].shape)   # (32, 50, 50, 3) (32,)

# np.save('./_save_npy/keras48_1_train_x.npy', arr = xy_train[0][0])
# np.save('./_save_npy/keras48_1_train_y.npy', arr = xy_train[0][1])
# np.save('./_save_npy/keras48_1_test_x.npy', arr = xy_test[0][0])
# np.save('./_save_npy/keras48_1_test_y.npy', arr = xy_test[0][1])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(xy_train, epochs = 20, steps_per_epoch = 801, 
                    validation_data = xy_test,
                    validation_steps = 4,)


model.save_weights('./_save/keras48_1_save_weights.h5')



acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

model.evaluate_generator


'''
loss: 8.121248899067846e-14
val_loss: 9.126583690190335e-33
acc: 1.0
val_acc: 1.0

https: // www.kaggle.com/c/dogs-vs-cats/data
'''