# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해짐
# 그리고 Conv2D(32, ...) 라면 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 32개가 되도록 만든다는 뜻
# 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 32개가 추가로 더해짐
# ex) 3 x 3(필터 크기) x 3 (입력 채널(RGB, 흑백이면 1)) x 32(#출력 채널) + 32(출력 채널 bias) = 896

# model.add(Conv2D(a, kernel_size=(b,c), input_shape=(d, e, f)))
# a = 출력채널
# Filter와 Kernel은 같음 ex) (b,c) -> kernel_size
# d, e, 
# f = channel : 컬러 이미지는 3개의 채널로 구성됨. 반면에 흑백 명암만을 표현하는 흑백 사진은 2차원 데이터로 1개 채널로 구성됨 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))   # 9, 9, 10  파라미터 수 : (2x2(커넬사이즈)+1(바이어스))X10(채널)
model.add(Conv2D(5, (3,3), activation='relu'))                      # 7, 7, 5   파라미터 수 : ((3X3+1))X5(채널)X10(전 채널)
model.add(Conv2D(7, (2,2), activation='relu'))                      # 6, 6, 7
model.add(Flatten())                                                # 평평하게 핌(1줄로)
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))

model.summary()


# Output Size = (W - F + 2P) / S + 1
# W: input_volume_size
# F: kernel_size
# P: padding_size
# S: strides
