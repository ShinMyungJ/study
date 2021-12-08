# 파라미터의 수


# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = 필터
# Filter와 Kernel은 같음 ex) (b,c) -> kernel_size
# d = row, e = col 
# f = channel : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

#padding kernel_size=(2,2)이면 위,좌 한칸 (3,3)이면 상하좌우 모두 감싸줌

# model.add(Dense(w)) w = units

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers.core import Dropout

model = Sequential()                  # strides 한번에 n만큼 이동하여 kernel_size가 이동하여 자름
model.add(Conv2D(10, kernel_size=(2,2), strides=1,           # padding 감싸줘서 1번만 중첩되는 것 방지 same <> valid
                 padding='same', input_shape=(10, 10, 1)))   # 9, 9, 10  파라미터 수 : (((2 x 2 ) + 1)) X 10
# model.add(MaxPooling2D(3))                                   # MaxPooling : 풀사이즈에서 가장 큰 값만 사용. shape는 반(defult 2)
                                                             #                        ((kernal_size))+(bias))X필터
model.add(Conv2D(5, (3,3), activation='relu'))               # 7, 7, 5   파라미터 수 : (3 X 3) X 5 X 10 + 1 X 5
                                                             #                        (kernal_size)X(전 필터)X(필터)+(bias)X필터
model.add(Conv2D(7, (2,2), activation='relu'))             # 6, 6, 7
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
