# y = wx + b
import tensorflow as tf
tf.set_random_seed(77)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]

# W = tf.Variable(2, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(W))                   # random seed 66 일때, [0.06524777] 77일때, [1.014144]


#2. 모델 구성
hypothesis = x_train * W + b                # model.add(Dense())

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)    # 여기서 가중치 갱신해줌

train = optimizer.minimize(loss)            # train만 sess.run 하면 다른 연산이 모두 가능, 모두 연결되어 있기 때문!
# optimizer = 'sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
# Launch the graph in a session
with tf.compat.v1.Session() as sess:      # sess = tf.Session() with문이 끝나는 순간 자동으로 close가 됨(세션 닫기용)

    # sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))
        
# sess.close()
