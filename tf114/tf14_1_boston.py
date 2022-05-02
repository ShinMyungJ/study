from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)
print(x_data.shape, y_data.shape)     # (506, 13) (506, )

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape)     # (354, 13) (354, 1)
print(x_test.shape, y_test.shape)       # (152, 13) (152, 1)

w = tf.Variable(tf.random.normal([13, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1]), name='bias')

#2. 모델 구성
# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(50001):
    _, loss_v, w_v , b_v = sess.run([train, loss, w, b], feed_dict={x:x_train, y:y_train})
    # print(epochs, '\t', loss_v, '\t' , w_v, '\t', b_v)
    if epochs % 1000 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(epochs, loss_v)


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = tf.matmul(x, w_v) + b_v
y_predict_data = sess.run(y_predict, feed_dict={x: x_test})
# print(y_pred)
r2 = r2_score(y_test, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_test, y_predict_data)
print("mae : ", mae)
    
sess.close()

# r2 :  0.740261484529011
# mae :  3.253609189120206

# r2 :  0.8057474930217513
# mae :  3.019353417346352