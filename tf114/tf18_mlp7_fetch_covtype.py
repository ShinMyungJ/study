from sklearn.datasets import fetch_covtype, load_wine
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)
sess = tf.compat.v1.Session()

#1. 데이터
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target
print(np.unique(y_data))
y_data = sess.run(tf.one_hot(y_data, depth=7))
print(x_data.shape, y_data.shape)     # (581012, 54) (581012, 7)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])       
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=66)
print(x_train.shape, y_train.shape)     # (406708, 54) (406708, 7)
print(x_test.shape, y_test.shape)       # (174304, 54) (174304, 7)

#2. 모델 구성
w1 = tf.compat.v1.Variable(tf.zeros([54,10]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([10]), name="bias1")
Hidden_layer1 = tf.matmul(x, w1) + b1             # bias도 y의 칼럼 수 만큼 구성

w2 = tf.compat.v1.Variable(tf.zeros([10,50]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([50]), name='bias2')
Hidden_layer2 = tf.nn.relu(tf.matmul(Hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([50,30]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([30]), name='bias3')
Hidden_layer3 = tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([30,15]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([15]), name='bias4')
Hidden_layer4 = tf.matmul(Hidden_layer3, w4) + b4 

w5 = tf.compat.v1.Variable(tf.random.normal([15,7]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([7]), name='bias5')

hypothesis = tf.nn.softmax(tf.matmul(Hidden_layer4, w5) + b5)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))           # categorical_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
# train = optimizer.minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

#3-2. 훈련
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(401):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0:
            print(step, loss_val)
            
    # predict
    results = sess.run(hypothesis, feed_dict={x:x_test})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # print(results.shape)
    
    acc = accuracy_score(sess.run(tf.math.argmax(y_test,1)), sess.run(tf.math.argmax(results, 1)))
    print('acc : ', acc)
    print('loss : ', loss_val)
    
   

# sess.close()

# acc :  0.48640880301083167
