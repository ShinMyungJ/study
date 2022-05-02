from sklearn.datasets import load_diabetes
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)     # (442, 10) (442,)
y_data = y_data.reshape(442, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([10, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1]), name='bias')

#2. 모델 구성
# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-4)
# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(50001):
    _, loss_v, w_v , b_v = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
    # print(epochs, '\t', loss_v, '\t' , w_v, '\t', b_v)
    if epochs % 1000 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(epochs, loss_v)


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = tf.matmul(x, w_v) + b_v
y_predict_data = sess.run(y_predict, feed_dict={x: x_data})
# print(y_pred)
r2 = r2_score(y_data, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_data, y_predict_data)
print("mae : ", mae)
    
sess.close()

# r2 :  0.22774428950364267
# mae :  57.91934620093436