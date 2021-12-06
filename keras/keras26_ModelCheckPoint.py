from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(65))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))
# model.summary()

# model.save("./_save/keras25_1_save_model.h5")
# model.save_weights("./_save/keras25_1_save_weights.h5")
# model.load_weights("./_save/keras25_1_save_weights.h5")
# loss :  9405.05859375
# r2 스코어 :  -111.52366952115946
# model.load_weights("./_save/keras25_3_save_weights.h5")


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                               # 컴파일까지 명시해줘야 한다

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=1) #, restore_best_weights=True)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1, save_best_only=True,
                      filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, callbacks=[es, mcp])
end = time.time() - start

model.save("./_save/keras26_1_save_model.h5")

print("=================================================")
print(hist)
print("=================================================")
print(hist.history)
print("=================================================")
print(hist.history['loss'])
print("=================================================")
print(hist.history['val_loss'])
print("=================================================")

import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')          # 선을 긋는다
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()                  # 점자 형태
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

print("걸린시간 : ", round(end, 3), '초')

# model.save_weights("./_save/keras25_3_save_weights.h5")                             # fit 다음에 model.save하면 model과 weight까지 저장된다
# model.load_weights("./_save/keras25_3_save_weights.h5")
# model.load_weights("./_save/keras25_1_save_weights.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  35.21707534790039
# r2 스코어 :  0.5786570847910619