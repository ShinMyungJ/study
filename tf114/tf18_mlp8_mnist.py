from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)
sess = tf.compat.v1.Session()

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
print(np.unique(y_train))
y_train = sess.run(tf.one_hot(y_train, depth=10))       # (60000, 784) (60000, 10)
y_test = sess.run(tf.one_hot(y_test, depth=10))         # (10000, 784) (10000, 10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])       
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델 구성
w1 = tf.compat.v1.Variable(tf.zeros([x_train.shape[1], 64]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([64]), name="bias1")
Hidden_layer1 = tf.matmul(x, w1) + b1             # bias도 y의 칼럼 수 만큼 구성

w2 = tf.compat.v1.Variable(tf.zeros([64,32]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([32]), name='bias2')
Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([32,16]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([16]), name='bias3')
Hidden_layer3 = tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([16,10]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias4')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer3, w4) + b4)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))           # categorical_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
# train = optimizer.minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=2e-3).minimize(loss)

#3-2. 훈련
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 10 == 0:
            print(step, loss_val)
            
    # predict
    results = sess.run(hypothesis, feed_dict={x:x_test})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # print(results.shape)
    
    acc = accuracy_score(sess.run(tf.math.argmax(y_test,1)), sess.run(tf.math.argmax(results, 1)))
    print('acc : ', acc)
    print('loss : ', loss_val)
    
   
# sess.close()

# acc :  0.1135
# loss :  2.301166
