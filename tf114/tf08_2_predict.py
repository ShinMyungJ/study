# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

# 위 값들을 이용해서 predict해라.
# x_test 라는 placeholder 생성

# y = wx + b
import tensorflow as tf
tf.set_random_seed(77)

#1. 데이터
# x_train = [1,2,3]
# y_train = [1,2,3]
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

# W = tf.Variable(2, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)


#2. 모델 구성
hypothesis = x_train * W + b                # model.add(Dense())
# hypothesis2 = x_test * W + b                # model.add(Dense())

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse
# loss = tf.reduce_mean(tf.square(hypothesis - y_test))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)    # 여기서 가중치 갱신해줌

train = optimizer.minimize(loss)            # train만 sess.run 하면 다른 연산이 모두 가능, 모두 연결되어 있기 때문!
# optimizer = 'sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line
for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                            feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)
        

#################### 실습, 과제 ###############################

x_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val

print("[6, 7, 8] 예측 : ", sess.run(y_predict, feed_dict={x_test: x_data}))
# print(sess.run(hypothesis2, feed_dict = {x_test: [4]}))      
# print(sess.run(hypothesis2, feed_dict = {x_test: [5, 6]}))      
# print(sess.run(hypothesis2, feed_dict = {x_test: [6, 7, 8]}))      


sess.close()
