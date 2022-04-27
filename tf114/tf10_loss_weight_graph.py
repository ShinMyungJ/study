from matplotlib import RcParams, rcParams
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

print('버전 : ', mpl.__version__)
print('설치 위치 : ', mpl.__file__)
print('설정 위치 : ', mpl.get_configdir())
print('캐시 위치 : ', mpl.get_cachedir)

x = [1,2,3]
y = [1,2,3]
w = tf.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))                     # cost, error, loss 같은 말

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("=================== W history ==================")
print(w_history)
print("=================== loss history ==================")
print(loss_history)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumGothic'
plt.xlabel('웨이트')
plt.ylabel('로스')
plt.title('슨상님 10000세')
plt.show()



