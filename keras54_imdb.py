from tensorflow.keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

# print(x_train.shape, x_test.shape)      # (25000,) (25000,)
# print(y_train.shape, y_test.shape)      # (25000,) (25000,)
# print(np.unique(y_train))               # [0 1]

# print(x_train[0], y_train[0])

# print(type(x_train), type(y_train))         # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(x_train.shape, y_train.shape)         # (8982,) (8982,)

# print(len(x_train[0]), len(x_train[1]))     # 218 189
# print(type(x_train[0]), type(x_train[1]))   # <class 'list'> <class 'list'>

# print("최대길이 : ", max(len(i) for i in x_train))             # 최대길이 :  2494
# print("평균길이 : ", sum(map(len, x_train)) / len(x_train))    # 평균길이 :  238.71364

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=200, truncating='pre')

print(x_train.shape, y_train.shape)        # (25000, 200) (25000,)
print(x_test.shape, y_test.shape)          # (25000, 200) (25000,)

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
import time


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=300, input_length=200))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'       
model_path = "".join([filepath, 'k54_imdb_', datetime, '_', filename])

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

