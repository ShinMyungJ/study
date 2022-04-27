import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)
print(tf.executing_eagerly())   # False

# 즉시실행모드!!!
tf.compat.v1.disable_eager_execution()  #즉시 실행모드 꺼!라는 뜻  # tensor2 환경에서 sess.run을 사용할 수 있게 만듦

print(tf.executing_eagerly())

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))


