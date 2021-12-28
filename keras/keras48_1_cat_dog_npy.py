# 세이브 한뒤에
# 세이브 한 소스는 주석처리
# 로드해서 처리

# kaggle.com/c/dogs-vs-cats/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest'
# )
# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )

# # D:\_data\image\brain

# xy_train = train_datagen.flow_from_directory(
#     '../_data/image/cat_dog/training_set',
#     target_size=(150, 150),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
#     batch_size=8100,
#     class_mode='binary',
#     shuffle=True
# )       # Found 8005 images belonging to 2 classes.

# xy_test = test_datagen.flow_from_directory(
#     '../_data/image/cat_dog/test_set',
#     target_size=(150,150),
#     batch_size=2100,
#     class_mode='binary'    
# )       # Found 2023 images belonging to 2 classes.

# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001D297074F40>

# np.save('./_save_npy/keras48_1_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras48_1_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras48_1_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras48_1_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras48_1_train_x.npy')
y_train = np.load('./_save_npy/keras48_1_train_y.npy')
x_test = np.load('./_save_npy/keras48_1_test_x.npy')
y_test = np.load('./_save_npy/keras48_1_test_y.npy')

print(x_train)
print(x_train.shape)

#2. 모델 구성하시오!!!

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(2))
model.add(Conv2D(16, (2,2)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
# model.fit(xy_train[0][0], xy_train[0][1])

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k48_1_load_', datetime, '_', filename])

es = EarlyStopping(monitor='val_accuracy', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_accuracy', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_acc[-1])

#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/MJ/'
sample_image = sample_directory + "MJ.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

print("-- Evaluate --")
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(150, 150, 3))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
elif(classes[0][0]>=0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
else:
    print("ERROR")

# -- Predict --
# [[0.7477025]]
# 당신은 74.77 % 확률로 개 입니다

# 걸린시간 :  512.287 초
# loss :  0.14823612570762634
# val_loss :  2.2141449451446533
# accuracy :  0.942535936832428
# val_accuracy :  0.5346658229827881