import tensorflow as tf
tf.compat.v1.set_random_seed(66)
                        # input_dim = 1
변수 = tf.compat.v1.Variable(tf.random.normal([1]), name='weight')
print(변수)

#1. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)           # tf형을 사람형으로 변환
print('aaa : ', aaa)        # [0.06524777]
sess.close()

#2.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)
print('bbb : ', bbb)
sess.close()

#3.
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc : ', ccc)
sess.close()
