# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(2, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델 구성
hypothesis = x_train * W + b                # model.add(Dense())

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)    # 여기서 가중치 갱신해줌
'''
learning_rate = 0.1 # 학습률
gradient = tf.reduce_mean((W * X - Y) * X) # d/dW
descent = W - learning_rate * gradient #경사하강법
update = W.assign(descent) # 업데이트
'''
train = optimizer.minimize(loss)            # train만 sess.run 하면 다른 연산이 모두 가능, 모두 연결되어 있기 때문!
# optimizer = 'sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
# Launch the graph in a session
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
        
sess.close()
