import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 80, 75],                         # (5, 3)
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]

y_data = [[152], [185], [180], [196], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])       # n, 3 <- 3은 w의 행
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])       # n, 1 <- 1은 w의 열

w = tf.Variable(tf.random.normal([3, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
# optimizer = tf.train.AdamOptimizer(learning_rate=4e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(70001):
    _, loss_v, w_v , b_v = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
    # print(epochs, '\t', loss_v, '\t' , w_v, '\t', b_v)
    if epochs % 1000 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(epochs, loss_v, w_v[0][0],w_v[1][0],w_v[2][0], b_v)

from sklearn.metrics import r2_score, mean_absolute_error

y_predict = tf.matmul(x, w_v) + b_v
y_predict_data = sess.run(y_predict, feed_dict={x: x_data})
# print(y_pred)
r2 = r2_score(y_data, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_data, y_predict_data)
print("mae : ", mae)
    
sess.close()

# Adam
# r2 :  0.9997270494853739
# mae :  0.3134674072265625

# r2 :  0.9996362453297368
# mae :  0.321002197265625

# r2 :  0.9995678495205169
# mae :  0.3292327880859375