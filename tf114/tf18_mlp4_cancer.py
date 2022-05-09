from sklearn.datasets import load_breast_cancer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)
sess = tf.compat.v1.Session()

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)
print(x_data.shape, y_data.shape)     # (569, 30) (569, 1)

x = tf.compat.v1.placeholder(tf.float64, shape=[None, 30])       
y = tf.compat.v1.placeholder(tf.float64, shape=[None, 1])

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,train_size = 0.7,random_state=66)
print(x_train.shape, y_train.shape)     # (398, 30) (398, 1)
print(x_test.shape, y_test.shape)       # (171, 30) (171, 1)
y_train = y_train.astype(float)
y_test = y_test.astype(float)
print(x_train.dtype, y_train.dtype)
print(x.dtype, y.dtype)

#2. 모델 구성
w1 = tf.compat.v1.Variable(tf.zeros([30,5]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([5]), name="bias1")
Hidden_layer1 = tf.matmul(x, w1) + b1             # bias도 y의 칼럼 수 만큼 구성

w2 = tf.compat.v1.Variable(tf.random.uniform([5,20]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([20]), name='bias2')
Hidden_layer2 = tf.matmul(Hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([20,10]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias3')
Hidden_layer3 = tf.nn.relu(tf.matmul(Hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([10,1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias4')
# Hidden_layer4 = tf.matmul(Hidden_layer3, w4) + b4 

# w5 = tf.compat.v1.Variable(tf.random.normal([10,3]), name='weight5')
# b5 = tf.compat.v1.Variable(tf.random.normal([3]), name='bias5')

hypothesis = tf.sigmoid(tf.matmul(Hidden_layer3, w4) + b4)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # sigmoid
# loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
optimizer = tf.train.AdamOptimizer(learning_rate=0.000005).minimize(loss)
# train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(300):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, optimizer],
                                   feed_dict={x:x_train, y:y_train})
    if epochs % 20 == 0:
        print(epochs, 'loss : ', loss_val)


#4. 평가, 예측
from sklearn.metrics import accuracy_score, mean_absolute_error

y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float64)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, y_predict), dtype=tf.float64))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_test, y:y_test})

print("====================================")
print("예측값 : \n", hy_val[:5])
print("예측결과 : \n ", pred[:5])
print("Accuracy : ", acc)
print("loss : ", loss_val)
    
sess.close()

# Accuracy :  0.92355007
# loss :  0.24122475
