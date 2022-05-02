import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

#2. 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])       
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([2, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# model.add(Dense(1, activation ='sigmoid'))

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'loss : ', loss_val, '\n', hy_val)


#4. 평가, 예측

from sklearn.metrics import accuracy_score, mean_absolute_error

y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# print(y_predict)        # Tensor("Cast:0", shape=(?, 1), dtype=float32)
# print(sess.run(hypothesis > 0.5, feed_dict={x:x_data, y:y_data}))[[False]
# [[False]
#  [False]
#  [False]
#  [ True]
#  [ True]
#  [ True]]


# tf.cast
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
# 부동소수점형에서 정수형으로 바꾼 경우 소수점 버림을 한다.
# Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("====================================")
print("예측값 : \n", hy_val)
print("예측결과 : \n ", pred)
print("Accuracy : ", acc)

'''
y_predict_data = sess.run(y_predict, feed_dict={x: x_data})
print(y_predict_data)
acc = accuracy_score(y_data, y_predict_data)
print('acc : ', acc)

mae = mean_absolute_error(y_data, y_predict_data)
print("mae : ", mae)
'''    
sess.close()
