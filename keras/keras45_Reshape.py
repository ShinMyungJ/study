from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers.recurrent import LSTM

model = Sequential()                    # strides : image에 filter를 적용할 때 filter의 이동량, default : 1 
                              # input_shape(4,4) * kernel_size(2,2) 일때 stride=1이면 (3,3)/ 2라면 (2,2) CNN에서 주로 1로 설정
model.add(Conv2D(10, kernel_size=(2,2), strides=1,         # padding : default = valid(비사용) / same(사용) 
                 padding='same', input_shape=(28, 28, 1)))   
model.add(MaxPooling2D(2))
model.add(Conv2D(5, (2,2), activation='relu'))            # 13, 13, 5
model.add(Conv2D(7, (2,2), activation='relu'))            # 12, 12, 7
model.add(Conv2D(7, (2,2), activation='relu'))            # 11, 11, 7
model.add(Conv2D(10, (2,2), activation='relu'))           # 10, 10, 10
# model.add(Flatten())                                    # (N, 1000)       평평하게 핌(1줄로) 이미지 형태의 데이터를 배열 형태로 만듦
model.add(Reshape(target_shape=(100,10)))                 # (N, 100, 10) 'target_shape =' 생략 가능
model.add(Conv1D(5,2))
model.add(LSTM(15))
model.add(Dense(10, activation="softmax"))

# model.add(Conv2D(5, (2,2)))
# model.add(Dense(64))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation='softmax'))

model.summary()
