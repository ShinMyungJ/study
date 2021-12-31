from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train, len(x_train), len(x_test))            # 8982, 2246
print(y_train[0])
print(np.unique(y_train))       # 46개의 뉴스종류
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train))         # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape)         # (8982,) (8982,)

print(len(x_train[0]), len(x_train[1]))     # 87, 56
print(type(x_train[0]), type(x_train[1]))   # <class 'list'> <class 'list'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))             # 뉴스기사의 최대길이 :  2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))    # 뉴스기사의 평균길이 :  145.5398574927633
                                                                        # map은 x_train이라는 리스트의 len을 반환해주는 역할
# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
# print(x_train.shape)            # (8982, 2376) -> (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)        # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)          # (2246, 100) (2246, 46)

######################################################################################

word_to_index = reuters.get_word_index()
# print(word_to_index)
# print(sorted(word_to_index.items()))  # 키 위주로 나옴
import operator
print(sorted(word_to_index.items(), key = operator.itemgetter(1)))      # itemgetter(0) key중심, itemgetter(1) value중심

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in x_train[0]]))


#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
import time

# 실습 시작!!! 완성

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=300, input_length=100))
model.add(LSTM(16))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       
model_path = "".join([filepath, 'k53_reuters_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es,mcp])

end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')


#4. 평가, 훈련
scores = model.evaluate(x_test, y_test)
print("%s: %.2f" %(model.metrics_names[0], scores[0]))
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 걸린시간 :  122.879 초
# 71/71 [==============================] - 0s 5ms/step - loss: 1.3494 - acc: 0.6897
# loss: 1.35
# acc: 68.97%

# 걸린시간 :  129.375 초
# 71/71 [==============================] - 0s 6ms/step - loss: 1.6367 - acc: 0.6224
# loss: 1.64
# acc: 62.24%

# {key : value} 사회통념상 자주 사용하는 것