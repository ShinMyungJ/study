from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target.reshape(-1, 1)
print(x_data.shape, y_data.shape)     # (506, 13) (506,1)
# y_data = y_data.reshape(506, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성
# Input layer
w1 = tf.compat.v1.Variable(tf.random.uniform([13,50]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([50]), name="bias1")
Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)
# Hidden_layer1 = tf.matmul(x, w1) + b1
# Hidden_layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([50,30]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([30]), name='bias2')
Hidden_layer2 = tf.matmul(Hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([30,15]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([15]), name='bias3')
Hidden_layer3 = tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([15,8]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([8]), name='bias4')
Hidden_layer4 = tf.matmul(Hidden_layer3, w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([8,1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias5')

hypothesis = tf.matmul(Hidden_layer4, w5) + b5

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
# train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import r2_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0:
            print(step, loss_val)
            
    # predict
    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # print(results.shape)
    
    r2 = r2_score(y_test, y_predict)
    print('r2 : ', r2)
    print('loss : ', loss_val)


    
# sess.close()

# r2 :  
# mae :  