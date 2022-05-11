import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import to_categorical # 1부터 시작한다.
# one hot은 0부터
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128]) # Conv2D에서 kernel size 역할을 한다.
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding= 'SAME')

# model.add(conv2D(filters = 64, kernel_size = (2,2), strides=(1,1), padding = 'SAME', input_shape = (28, 28, 1))) # 위의 두줄과 같다.

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 28, 28, 64), dtype=float32)
print(L1_maxpool) # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

# Layer 2
w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L2_maxpool)       # (?, 7, 7, 64)

# Layer 3
w3 = tf.compat.v1.get_variable('w3', shape = [3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1,1,1,1], padding = 'SAME')
L3 = tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L3_maxpool)       # (?, 4, 4, 64)

# Layer 4
w4 = tf.compat.v1.get_variable('w4', shape = [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides = [1,1,1,1], padding = 'SAME')
L4 = tf.nn.selu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L4_maxpool)       # (?, 4, 4, 64)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*32]) # 플래튼 :  Tensor("Reshape:0", shape=(?, 128), dtype=float32)
print("플래튼 : ", L_flat)

# layer 5 DNN

w5 = tf.compat.v1.Variable(tf.random.uniform([2*2*32,64]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.uniform([64]), name='bias5')
Hidden_layer1 = tf.matmul(L_flat, w2) + b2

w6 = tf.compat.v1.Variable(tf.random.normal([64,32]), name='weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([32]), name='bias6')
Hidden_layer2 = tf.matmul(Hidden_layer1, w6) + b6

w7 = tf.compat.v1.Variable(tf.random.normal([32,10]), name='weight7')
b7 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias7')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer2, w7) + b7)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))           # categorical_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
# train = optimizer.minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

#3-2. 훈련
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0:
            print(step, loss_val)
            
    # predict
    results = sess.run(hypothesis, feed_dict={x:x_test})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # print(results.shape)
    
    acc = accuracy_score(sess.run(tf.math.argmax(y_test,1)), sess.run(tf.math.argmax(results, 1)))
    print('acc : ', acc)
    print('loss : ', round(4, loss_val))