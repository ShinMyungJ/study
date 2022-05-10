import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2,30]), name='weight1')
b1 = tf.compat.v1.Variable(tf.random.normal([30]), name='bias1')

Hidden_layer1 = tf.sigmoid(tf.matmul(x, w1), b1)
layers = tf.nn.dropout(Hidden_layer1, keep_prob=0.7)