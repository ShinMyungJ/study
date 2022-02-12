import numpy as np
import autokeras as ak
import tensorflow as tf


#1. 데이터

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    # tf.keras.datasets.mnist.load_data()


#2. 모델
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=5
)

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train, epochs=10)
end = time.time() - start

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)
print('걸린시간 : ', round(end, 4), '초')

# model.summary()

# [2.6375105381011963, 0.34040001034736633]