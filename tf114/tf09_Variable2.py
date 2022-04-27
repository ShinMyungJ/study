
###################################### 1. Session() // sess.run(변수)

# y = wx + b
import tensorflow as tf
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(2, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)


#2. 모델 구성
hypothesis = x_train * W + b                # model.add(Dense())

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.167)    # 여기서 가중치 갱신해줌

train = optimizer.minimize(loss)            # train만 sess.run 하면 다른 연산이 모두 가능, 모두 연결되어 있기 때문!
# optimizer = 'sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line
for step in range(201):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                            feed_dict={x_train:x_train_data, y_train:y_train_data})

#################### predict

x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val

print("[6, 7, 8] 예측 : ", sess.run(y_predict, feed_dict={x_test: x_test_data}))

sess.close()


###################################### 2. Session() // 변수.eval(session=sess)

# y = wx + b
import tensorflow as tf
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)


#2. 모델 구성
hypothesis = x_train * W + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.167)    # 여기서 가중치 갱신해줌

train = optimizer.minimize(loss)           

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line
for step in range(201):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                            feed_dict={x_train:x_train_data, y_train:y_train_data})

#################### predict
      

x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val

print("[6, 7, 8] 예측 : ", y_predict.eval(session=sess, feed_dict={x_test: x_test_data}))

sess.close()


###################################### 3. InteractiveSession() // 변수.eval()


# y = wx + b
import tensorflow as tf
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(2, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)


#2. 모델 구성
hypothesis = x_train * W + b                # model.add(Dense())

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.167)    # 여기서 가중치 갱신해줌

train = optimizer.minimize(loss)            # train만 sess.run 하면 다른 연산이 모두 가능, 모두 연결되어 있기 때문!
# optimizer = 'sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line
for step in range(201):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                            feed_dict={x_train:x_train_data, y_train:y_train_data})
        

#################### predict

x_test_data = [6, 7, 8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * W_val + b_val

print("[6, 7, 8] 예측 : ", y_predict.eval(feed_dict={x_test: x_test_data}))

sess.close()