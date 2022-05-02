import numpy as np
import tensorflow as tf

tf.set_random_seed(66)

#1. 데이터
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_predict = [[1, 11, 7, 9]]         # (1, 4) -> (N, 4)

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])       
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1, 3]), name='bias')          # bias도 y의 칼럼 수 만큼 구성

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)                 # nn은 뉴럴네트워크
# model.add(Dense(3, activation ='softmax'))

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))       # binary_crossentropy

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))           # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, loss_val)
            
    # predict
    results = sess.run(hypothesis, feed_dict={x:x_data})
    print(results, sess.run(tf.arg_max(results, 1)))
    print(results.shape)
    
    acc = accuracy_score(sess.run(tf.arg_max(y_data,1)), sess.run(tf.arg_max(results, 1)))
    print('acc : ', acc)

    # mae = mean_absolute_error(y_data, y_predict_data)
    # print("mae : ", mae)
        
    



   
# sess.close()


