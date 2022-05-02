import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(66)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run(w))        # [-0.3266699]

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.21
gradient = tf.reduce_mean((x * w - y) * x)
descent = w - lr * gradient
update = w.assign(descent)      # w = w - lr * gradient

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

feed_dict = {x:x_train, y:y_train}

for step in range(6):
    # sess.run(update, feed_dict=feed_dict)
    # print(step, '\t', 
    # update_eval = update.eval(feed_dict=feed_dict)
    # print(step, '\t', update_eval, w.eval())
    
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict=feed_dict)
    print(step, '\t', loss_v, '\t' , w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
    
sess.close()

print("=================== W history ==================")
print(w_history)
print("=================== loss history ==================")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weight')
plt.ylabel('loss')
plt.show()




