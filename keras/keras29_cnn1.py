# 파라미터의 수
# (3, 3) 필터 한개에는 3 x 3 = 9개의 파라미터가 있음(Numpy연산 방식으로 이해)
# 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 3이 곱해짐
# 그리고 Conv2D(32, ...) 라면 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 32개가 되도록 만든다는 뜻
# 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 32개가 추가로 더해짐
# ex) 3 x 3(필터 크기) x 3 (#입력 채널(RGB)) x 32(#출력 채널) + 32(출력 채널 bias) = 896

# model.add에 딸린 값 정식명칭
# model.add(Conv2D(5, (3,3), activation='relu'))
# 5 -> filters,  (3,3) -> kernel_size

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))   # 9, 9, 10
model.add(Conv2D(5, (3,3), activation='relu'))                      # 7, 7, 5
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



