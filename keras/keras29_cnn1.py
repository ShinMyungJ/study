# CNN : Convolution Neural Network(합성곱 신경망), 시신경 구조를 모방, 이미지 데이터에 주로 사용됨

# 파라미터의 수

# Layer (type)                 Output Shape              Param #
# ================================================================= 파라미터 수 : (((2 x 2 ) + 1)) X 10 = 50
# conv2d (Conv2D)              (None, 10, 10, 10)        50                   ((kernal_size))+(bias))X(필터)
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
# a = filters (Convolution filter의 수)
# Filter와 Kernel은 같음 ex) (b,c) -> b = Convolution Kernel's rows(행), c = Convolution Kernel's cols(열)
# d = rows(행, width), e = cols(열, height)
# f = channels : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

#padding kernel_size=(2,2)이면 위,좌 한칸 (3,3)이면 상하좌우 모두 감싸줌

# model.add(Dense(w)) w = units

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers.core import Dropout

model = Sequential()                    # strides : image에 filter를 적용할 때 filter의 이동량, default : 1 
                              # input_shape(4,4) * kernel_size(2,2) 일때 stride=1이면 (3,3)/ 2라면 (2,2) CNN에서 주로 1로 설정
model.add(Conv2D(10, kernel_size=(2,2), strides=1,         # padding : default = valid(비사용) / same(사용) 
                 padding='same', input_shape=(10, 10, 1)))   
model.add(MaxPooling2D(2)) # 2 = (2,2)                     # MaxPooling : 풀사이즈에서 가장 큰 값만 사용. default 2
model.add(Conv2D(5, (3,3), activation='relu'))             # MaxPooling은 Conv2D 다음 레이어에 적용해야함  
model.add(Conv2D(7, (2,2), activation='relu'))             
# model.add(Flatten())                                     # 평평하게 핌(1줄로) 이미지 형태의 데이터를 배열 형태로 만듦
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


# padding : input image에 convolution를 수행하면, output shape의 크기가 input shape의 크기보다 작아지게 됨
#           convolution를 거치면서 이미지는 작아지고 pixels의 정보는 점점 사라지게 되는 문제점을 해결하기 위해 이용
#           image의 가장자리에 특정 값으로 설정된 pixels를 추가하는 것을 zero-padding이라고 하며 CNN에서 주로 사용됨
# Maxpooling : CNN에서 주로 사용되며, 적당히 크기도 줄이고 특정 feature를 강조하는 역할을 Pooling layer에서 하게 됨
#              뉴런이 큰 신호에 반응하는 것과 유사. 노이즈가 감소! 속도 증가! 영상의 분별력 좋아짐!
#              (10,10) Maxpooling(2) 의 경우 (5,5) / Maxpooling(3) 의 경우 (3,3)이 되며 사용되지 못한 데이터는 소실
#              CNN이 처리해야하는 image의 Size가 크게 줄어들기 때문에 인공신경망의 model parameter 또한 크게 감소 
#              따라서, Pooling layer을 이용함으로써 CNN의 학습 시간을 크게 절약,
#              오버피팅 (overfitting) 문제 또한 어느정도 완화할 수 있음
# MaxPooling (3,3)
# model.add(MaxPooling2D(3))

# Fully Connected Layer
# Flatten Layer : 데이터 타입을 Fully Connected 네트워크 형태로 변경. 입력 데이터의 shape 변경만 수행
# Softmax Layer : Classification 수행

'''
Convolution Filter의 개수
각 Layer에서의 연산시간/량을 비교적 일정하게 유지하며 시스템의 균형을 맞추는 것이 좋다.
보통 Pooling Layer를 거치면 1/4로 출력이 줄어들기 때문에 Convolution Layer의 결과인 Feature Map의 개수를 4배정도 증가시키면 된다.

Filter 사이즈
작은 필터를 여러 개 중첩하면 원하는 특징을 더 돋보이게 하면서 연산량을 줄일 수 있다.
요즘 대부분의 CNN은 3x3 size를 중첩해서 사용한다고 한다.

Padding 여부
Padding은 Convolution을 수행하기 전, 입력 데이터 주변을 특정 픽셀 값으로 채워 늘리는 것이다.
Padding을 사용하게 되면 입력 이미지의 크기를 줄이지 않을 수 있다.

Stride
Stride는 Filter의 이동 간격을 조절하는 파라미터 이다.
이 값이 커지게 되면 결과 데이터의 사이즈가 작아지게 된다.

Pooling layer 종류
적당히 이미지 크기를 줄이면서 특정 feature를 강조하는 역할을 한다.
주로 Max 값을 뽑아내는 종류를 사용한다.
'''
