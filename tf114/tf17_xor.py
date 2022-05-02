import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# tf.random.uniform
# uniform(균일), uniform distribution(균등분포)
# tf.random.uniform은 원하는 형태의 랜덤 값을 가진 배열을 만듦.
# shape는 배열 형태, minval(최소값), maxval(최대값) 사이의 값을 반환함.

# tf.random.normal
# normal distribution(정규분포)
# tf.random.normal은 정규분포로 부터 주어진 형태와 자료형을 갖는 난수 텐서를 반환함.
# 첫번째 인자는 텐서, 두번째, 세번째 인자는 평균(mean), 표준편차(stddev)임.
# seed를 특정 정수값으로 지정하면 재사용 가능한 난수 텐서를 얻을 수 있음.

#2. 모델구성
# Input layer
w = tf.compat.v1.Variable(tf.random.normal([2,1]), name="weight1")
b = tf.compat.v1.Variable(tf.random.normal([1]), name="bias1")
# Hidden_layer1 = tf.matmul(x, w1) + b1
# Hidden_layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(5001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x:x_data, y:y_data})
    if epochs % 1000 == 0:
        print(epochs, 'loss : ', loss_val, '\n', hy_val)


#4. 평가, 예측

from sklearn.metrics import accuracy_score, mean_absolute_error

y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

sess.close()

print("====================================")
print("예측값 : \n", hy_val)
print("예측결과 : \n ", pred)
print("Accuracy : ", acc)