import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# node3 = node1 + node2
node3 = tf.add(node1, node2)

# node4 = node1 - node2
node4 = tf.subtract(node1, node2)

# node5 = node1 * node2
node5 = tf.multiply(node1, node2)

# node6 = node1 / node2
node6 = tf.divide(node1, node2)

# sess = tf.Session()
sess = tf.compat.v1.Session()

print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))

# 5.0
# -1.0
# 6.0
# 0.6666667

# 실습
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6