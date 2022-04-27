import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()        # 모든변수를 초기화 시킴(사용가능하게 바꿔줌)
# init = tf.compat.v1.variables_initializer([x, y])

sess.run(init)

print("잘나오니? ", sess.run(x))


