# 훈련데이터 10만개로 증폭
# 완료후 기존 모델과 비교
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 보고 삭제
'''
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    # vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 0.1,              
    # shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3       
    )             

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

batch_num = 800
xy_train = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human',
    target_size=(150,150),
    batch_size=batch_num,
    class_mode='binary',
    subset='training',
    shuffle=True
)      # Found 719 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/horse-or-human', # same directory as training data
    target_size=(150,150),
    batch_size=batch_num,
    class_mode='binary',
    subset='validation',
)      # Found 308 images belonging to 2 classes.


# 증폭
augment_size = 20000
randidx = np.random.randint(xy_train[0][0].shape[0], size = augment_size)
x_augmented = xy_train[0][0][randidx].copy()
y_augmented = xy_train[0][1][randidx].copy()

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],150,150,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],150,150,3)

# 증폭한 데이터 합침
x_augmented = train_datagen.flow(x_augmented,
                                 y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False,
                                 # save_to_dir="../_temp"
                                 ).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((xy_train[0][1], y_augmented))

# print(x_train.shape, y_train.shape)     # (3000, 150, 150, 3) (3000,)

#2. 모델 구성

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       # 2500(에포수)-0.3724(val_loss).hdf5
model_path = "".join([filepath, 'k50_6_horse_human_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1,
                 batch_size = 128, 
                 validation_split = 0.2, 
                 callbacks = [es,mcp])
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')
'''
#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import os

#이미지확인하기
nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

hor_dir = '../_data/image/horse-or-human/horses'
hu_dir = '../_data/image/horse-or-human/humans'

hor_dir_file_list = os.listdir(hor_dir)
hu_dir_file_list = os.listdir(hu_dir)
pic_num = 10
for i in range(pic_num):
    print(hor_dir_file_list[i])
for i in range(pic_num):
    print(hu_dir_file_list[i])
    
'''
next_horse_pix = os.path.join(hor_dir, fname) 
                  for fname in hor_dir:
                      [pic_index-8:pic_index]]
next_human_pix = [os.path.join(hu_dir, fname) for fname in [pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

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
image_ = image.load_img(str(sample_image), target_size=(150, 150, 3))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
xy_test.reset()
print(xy_test.class_indices)

if(classes[0][0]<=0.5):
    horse = 100 - classes[0][0]*100
    print(f"당신은 {round(horse,2)} % 확률로 horse 입니다")
elif(classes[0][0]>=0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 human 입니다")
else:
    print("ERROR")
'''