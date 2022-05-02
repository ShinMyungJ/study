import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(66)

x_train_data = [1, 2, 3]
y_train_data = [1, 2, 3]
x_test_data = [4, 5, 6]
y_test_data = [4, 5, 6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(w))        # [-0.3266699]

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.01
gradient = tf.reduce_mean((x * w - y) * x)
descent = w - lr * gradient
update = w.assign(descent)      # w = w - lr * gradient

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

feed_dict = {x:x_test_data, y:y_test_data}

for step in range(21):
    # sess.run(update, feed_dict=feed_dict)
    # print(step, '\t', 
    # update_eval = update.eval(feed_dict=feed_dict)
    # print(step, '\t', update_eval, w.eval())
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict=feed_dict)
    print(step, '\t', loss_v, '\t' , w_v)
    
    # w_history.append(w_v)
    # loss_history.append(loss_v)

# 맹그러바!!!

from sklearn.metrics import r2_score, mean_absolute_error

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict = x_test * w_v
y_predict_data = sess.run(y_predict, feed_dict={x_test: x_test_data})
# print(y_pred)
r2 = r2_score(y_test_data, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_test_data, y_predict_data)
print("mae : ", mae)
    
sess.close()



