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

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

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
    
sess.close()
