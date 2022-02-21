# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=0.5,
    zoom_range=0.3,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)

batch_num = 1800
xy_train = train_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size=(50, 50),
    batch_size=batch_num,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)       # Found 1764 images belonging to 3 classes.

xy_test = train_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size=(50, 50),
    batch_size=batch_num,
    class_mode='categorical',
    subset='validation'    
)       # Found 756 images belonging to 3 classes.

# print(xy_train[0][0])


# 증폭 데이터 생성
augment_size = 5000
randidx = np.random.randint(xy_train[0][0].shape[0], size = augment_size)
x_augmented = xy_train[0][0][randidx].copy()
y_augmented = xy_train[0][1][randidx].copy()

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],50,50,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],50,50,3)

# 증폭한 데이터 합침
x_augmented = train_datagen.flow(x_augmented,
                                 y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                 # save_to_dir="../_temp"
                                 ).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((xy_train[0][1], y_augmented))
net = DenseNet121(weights='imagenet', include_top=False,
              input_shape=(50,50,3))

print(x_train.shape, y_train.shape)     # (6764, 50, 50, 3) (6764, 3)

#2. 모델
model = Sequential()
model.add(net)
model.add(GlobalAveragePooling2D())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))



#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
lr = 2e-5
optimizer = Adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k70_3_rps_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100,
                 batch_size = 32, 
                 validation_split = 0.2, 
                 callbacks = [es,lr])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/MJ/'
sample_image = sample_directory + "ccc2.jpg"

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
image_ = image.load_img(str(sample_image), target_size=(50, 50, 3))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)  #NDIMS

print(classes)
xy_test.reset()
print(xy_test.class_indices)
# {'paper': 0, 'rock': 1, 'scissors': 2}
if(y_predict==0):
    paper = classes[0][0]*100
    print(f"이것은 {round(paper,2)} % 확률로 보 입니다")
elif(y_predict==1):
    rock = classes[0][1]*100
    print(f"이것은 {round(rock,2)} % 확률로 바위 입니다")
elif(y_predict==2):
    scissors = classes[0][2]*100
    print(f"이것은 {round(scissors,2)} % 확률로 가위 입니다")
else:
    print("ERROR")
    
    
# 걸린시간 :  416.47 초
# -- Evaluate --
# accuracy: 88.62%
# -- Predict --
# [[1.1906823e-01 6.8606729e-05 8.8086319e-01]]
# {'paper': 0, 'rock': 1, 'scissors': 2}
# 이것은 88.09 % 확률로 가위 입니다    
    