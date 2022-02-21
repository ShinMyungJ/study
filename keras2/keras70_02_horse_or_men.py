# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)

# D:\_data\image\horse-or-human\

xy_train = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True,
)       # Found 719 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',
    subset='validation'    
)       # Found 308 images belonging to 2 classes.

print(xy_train[0][0].shape, xy_test[0][1].shape)        # (10, 50, 50, 3) (10,)

# 증폭한 데이터 생성
augment_size = 5000
randidx = np.random.randint(xy_train[0][0].shape[0], size = augment_size)
x_augmented = xy_train[0][0][randidx].copy()
y_augmented = xy_train[0][1][randidx].copy()

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],50,50,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],50,50,3)

print(x_train.shape, x_test.shape)     # (10, 50, 50, 3) (10, 50, 50, 3)

# 증폭한 데이터 합침
x_augmented = train_datagen.flow(x_augmented,
                                 y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                 # save_to_dir="../_temp"
                                 ).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((xy_train[0][1], y_augmented))

print(x_train.shape, y_train.shape)     # (5010, 50, 50, 3) (5010,)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D

net = DenseNet121(weights='imagenet', include_top=False,
              input_shape=(50,50,3))

model = Sequential()
model.add(net)
model.add(GlobalAveragePooling2D())
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k70_01_cat_dog_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
reducelr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100,
                 batch_size = 32, 
                 validation_split = 0.2, 
                 callbacks = [es,reducelr]) #,mcp])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

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
scores = model.evaluate_generator(xy_test)
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
# {'horses': 0, 'humans': 1}
if(classes[0][0]<=0.5):
    horses = 100 - classes[0][0]*100
    print(f"당신은 {round(horses,2)} % 확률로 horse 입니다")
elif(classes[0][0]>0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 human 입니다")
else:
    print("ERROR")


# 걸린시간 :  314.649 초
# loss :  0.3621101975440979
# val_loss :  0.5035870671272278
# acc :  0.8331015110015869
# val_acc :  0.824999988079071

# 걸린시간 :  320.33 초
# -- Evaluate --
# acc: 50.65%
# -- Predict --
# [[0.13881768]]
# {'horses': 0, 'humans': 1}
# 당신은 86.12 % 확률로 horse 입니다

