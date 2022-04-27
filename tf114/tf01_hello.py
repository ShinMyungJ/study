import tensorflow as tf

print(tf.__version__)

# print('hello world')

hello = tf.constant("Hello World")

print(hello)

# sess = tf.Session()       # 최종 결과값을 뺄때 사용해야 함
sess = tf.compat.v1.Session()
print(sess.run(hello))

# tf.constant 상수(바뀌지 않는 수, 고정된 값)
# 상수는 full 대문자(네이밍 룰에 의해) ex) AAA, BBB, BABO

# tf.variable 변수(바뀌는 수)

# tf.placeholder

# 연산을 시키고 sess.run(op)를 통과시킨다.
# tensor 머신 사용하여 output 값 뽑아냄
# 그렇지 않으면 자료형만 프린트 됨