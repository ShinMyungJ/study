from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)
print(x_data.shape, y_data.shape)     # (569, 30) (569, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, shuffle=True, random_state=66)

w = tf.Variable(tf.zeros([30, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.zeros([1]), name='bias')

#2. 모델 구성
# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # sigmoid
# loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, optimizer],
                                   feed_dict={x:x_train, y:y_train})
    if epochs % 100 == 0:
        print(epochs, 'loss : ', loss_val)


#4. 평가, 예측
from sklearn.metrics import accuracy_score, mean_absolute_error

y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))

pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("====================================")
print("예측값 : \n", hy_val[:5])
print("예측결과 : \n ", pred[:20])
print("Accuracy : ", acc)
    
sess.close()

# Accuracy :  0.9525483