import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)        # 자동으로 float로 인식
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)

# sess = tf.Session()
sess = tf.compat.v1.Session()

print('node1, node2 : ', sess.run([node1, node2]))
print('node3 : ', sess.run(node3))