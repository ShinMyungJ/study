# 파라미터의 수

# Layer (type)                 Output Shape              Param #
# ================================================================= 파라미터 수 : (((2 x 2 ) + 1)) X 10 = 50
# conv2d (Conv2D)              (None, 10, 10, 10)        50                   ((kernal_size))+(bias))X필터
# ____________________________________ g__ h___i___________________  # g = new_rows, h = new_cols, i = filters
# conv2d_1 (Conv2D)            (None, 8, 8, 5)           455        파라미터 수 : (3 X 3) X 5 X 10 + 1 X 5 = 455
# _________________________________________________________________         (kernal_size)X(전 필터)X(필터)+(bias)X(필터)
# conv2d_2 (Conv2D)            (None, 7, 7, 7)           147
# ================================================================= 파라미터 수 : ((2 X 2) X 5 + 1) X 7 = 147
# Total params: 652                                                         ((kernal_size)X(전 필터)+(bias))X(필터)
# Trainable params: 652
# Non-trainable params: 0
# _________________________________________________________________

# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = filters
# Filter와 Kernel은 같음 ex) (b,c) -> b = filter의 rows, c = filter의 cols
# d = rows, e = cols 
# f = channels : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

#padding kernel_size=(2,2)이면 위,좌 한칸 (3,3)이면 상하좌우 모두 감싸줌

# model.add(Dense(w)) w = units

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers.core import Dropout

model = Sequential()                  # strides 한번에 n만큼 이동하여 kernel_size가 이동하여 자름
model.add(Conv2D(10, kernel_size=(2,2), strides=1,           # padding : default = valid(비사용) / same(사용) 
                 padding='same', input_shape=(10, 10, 1)))   
# model.add(MaxPooling2D(3))                                 # MaxPooling : 풀사이즈에서 가장 큰 값만 사용. shape는 반(defult 2)
model.add(Conv2D(5, (3,3), activation='relu'))               
model.add(Conv2D(7, (2,2), activation='relu'))             
# model.add(Flatten())                                       # 평평하게 핌(1줄로)
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation='softmax'))

model.summary()

# Output Size = (W - F + 2P) / S + 1
# W: input_volume_size
# F: kernel_size
# P: padding_size
# S: strides
