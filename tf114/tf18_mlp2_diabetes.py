from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
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

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, shuffle=True, random_state=66)

#2. 모델 구성
# Input layer
w1 = tf.compat.v1.Variable(tf.random.uniform([10,16]), name="weight1")
b1 = tf.compat.v1.Variable(tf.zeros([16]), name="bias1")
Hidden_layer1 = tf.matmul(x, w1) + b1
# Hidden_layer1 = tf.matmul(x, w1) + b1
# Hidden_layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([16,14]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.normal([14]), name='bias2')
Hidden_layer2 = tf.matmul(Hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([14,12]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([12]), name='bias3')
Hidden_layer3 = tf.matmul(Hidden_layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([12,10]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias4')
Hidden_layer4 = tf.matmul(Hidden_layer3, w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([10,8]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([8]), name='bias5')
Hidden_layer5 = tf.matmul(Hidden_layer4, w5) + b5

w6 = tf.compat.v1.Variable(tf.random.normal([8,6]), name='weight6')
b6 = tf.compat.v1.Variable(tf.random.normal([6]), name='bias6')
Hidden_layer6 = tf.matmul(Hidden_layer5, w6) + b6

w7 = tf.compat.v1.Variable(tf.random.normal([6,4]), name='weight7')
b7 = tf.compat.v1.Variable(tf.random.normal([4]), name='bias7')
Hidden_layer7 = tf.matmul(Hidden_layer6, w7) + b7

w8 = tf.compat.v1.Variable(tf.random.normal([4,2]), name='weight8')
b8 = tf.compat.v1.Variable(tf.random.normal([2]), name='bias8')
Hidden_layer8 = tf.matmul(Hidden_layer7, w8) + b8

w9 = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight9')
b9 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias9')

hypothesis = tf.matmul(Hidden_layer8, w9) + b9

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(50001):
    _, loss_v, w_v , b_v = sess.run([train, loss, w9, b9], feed_dict={x:x_train, y:y_train})
    # print(epochs, '\t', loss_v, '\t' , w_v, '\t', b_v)
    if epochs % 1000 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(epochs, loss_v)


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
# print(y_pred)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)
print('loss : ', loss_v)

mae = mean_absolute_error(y_test, y_predict)
print("mae : ", mae)
    
sess.close()

# r2 :  0.22774428950364267
# mae :  57.91934620093436