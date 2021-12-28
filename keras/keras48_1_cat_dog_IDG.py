# kaggle.com/c/dogs-vs-cats/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import math
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

batch_num = 10
xy_train = train_datagen.flow_from_directory(
    '../_data/image/cat_dog/training_set',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=batch_num,
    class_mode='binary',
    shuffle=True
)       # Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/cat_dog/test_set',
    target_size=(50, 50),
    batch_size=batch_num,
    class_mode='binary'    
)       # Found 2023 images belonging to 2 classes.

aaa = len(xy_train)
spe = math.ceil(aaa/batch_num)

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001D297074F40>

print(xy_train[31])       # 마지막 batch
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape, xy_train[0][1].shape)         # (10, 150, 150, 3), (10,)   # 흑백은 알아서 찾아라

print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32,(2,2), padding='same', input_shape = (50,50,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(16,(2,2)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

spe = len(xy_train)

date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       
model_path = "".join([filepath, 'k48_1_cat_dog_IDG_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=5, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit_generator(xy_train, epochs=4, steps_per_epoch=spe,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=xy_test,
                    validation_steps=4, callbacks=[es, mcp]
                    )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])


#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/MJ/'
sample_image = sample_directory + "MJ.jpg"

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('Off')
plt.show()

print("-- Evaluate --")
scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
xy_test.reset()
print(xy_test.class_indices)
# {'cats': 0, 'dogs': 1}
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
elif(classes[0][0]>=0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
else:
    print("ERROR")
    
    
# -- Predict --
# [[0.7261659]]
# {'cats': 0, 'dogs': 1}
# 당신은 72.62 % 확률로 개 입니다

