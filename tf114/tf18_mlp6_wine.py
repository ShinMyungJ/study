from sklearn.datasets import fetch_covtype, load_wine
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)
sess = tf.compat.v1.Session()

#1. 데이터
datasets = load_wine()
x_data = datasets.data
y_data = datasets.target
print(np.unique(y_data))
y_data = sess.run(tf.one_hot(y_data, depth=3))
print(x_data.shape, y_data.shape)     # (178, 13) (178, 3)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])       
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=66)
print(x_train.shape, y_train.shape)     # (124, 13) (124, 3)
print(x_test.shape, y_test.shape)       # (54, 13) (54, 3)

#2. 모델 구성
w1 = tf.compat.v1.Variable(tf.zeros([13,5]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([5]), name="bias1")
Hidden_layer1 = tf.matmul(x, w1) + b1             # bias도 y의 칼럼 수 만큼 구성

w2 = tf.compat.v1.Variable(tf.random.uniform([5,20]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([20]), name='bias2')
Hidden_layer2 = tf.matmul(Hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([20,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias3')
Hidden_layer3 = tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([10,10]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias4')
Hidden_layer4 = tf.matmul(Hidden_layer3, w4) + b4 

w5 = tf.compat.v1.Variable(tf.random.normal([10,3]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([3]), name='bias5')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer4, w5) + b5)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))           # categorical_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
# train = optimizer.minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=2e-6).minimize(loss)

#3-2. 훈련
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(20001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0:
            print(step, loss_val)
            
    # predict
    results = sess.run(hypothesis, feed_dict={x:x_test})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # print(results.shape)
    
    acc = accuracy_score(sess.run(tf.math.argmax(y_test,1)), sess.run(tf.math.argmax(results, 1)))
    print('acc : ', acc)
    print('loss : ', loss_val)
    
   

# sess.close()

# acc :  0.9629629629629629
# loss :  0.015455346
