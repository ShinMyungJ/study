from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
tf.set_random_seed(66)

#1. 데이터 
path = '../_data/kaggle/bike/'   
train = pd.read_csv(path+'train.csv')  
x = train.drop(['casual','registered','count'], axis=1)  
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x_data = x.drop('datetime', axis=1)
y_data = train['count']
y_data = np.log1p(y_data)
y_data = y_data.values.reshape(-1, 1)
# y_data = y_data.to_numpy.reshape(-1, 1)
# print(x_data.shape, y_data.shape)       # (10886, 8) (10886, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 12])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([12, 1]), name='weight')        # 행렬 곱의 shape (n, m) * (m, s) 의 shape = (n, s)
b = tf.Variable(tf.random.normal([1]), name='bias')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, shuffle=True, random_state=66)


#2. 모델 구성
# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))      # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(15001):
    _, loss_v, w_v , b_v = sess.run([train, loss, w, b], feed_dict={x:x_train, y:y_train})
    # print(epochs, '\t', loss_v, '\t' , w_v, '\t', b_v)
    if epochs % 100 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(epochs, loss_v)


#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error

y_predict = tf.matmul(x, w_v) + b_v
y_predict_data = sess.run(y_predict, feed_dict={x: x_data})
# print(y_pred)
r2 = r2_score(np.expm1(y_data), np.expm1(y_predict_data))
print('r2 : ', r2)

mae = mean_absolute_error(np.expm1(y_data), np.expm1(y_predict_data))
print("mae : ", mae)
    
sess.close()

'''
r2 = r2_score(y_test, y_pred)
print ('r2 :', r2)

rmse = RMSE(y_test,y_pred)
print('RMSE : ', rmse)

y_pred = np.expm1(y_pred)
print(y_pred[:5])
'''

# r2 :  0.45547771128359715
# mae :  0.8160765328886457