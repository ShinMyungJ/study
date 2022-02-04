import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)

# print(x.shape)
x_train_reshape = x_train.reshape(60000, 28*28)
print(x_train_reshape.shape)

# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

#####################################################
# 실습 
# pca를 통해 0.95 이상인 n_component가 몇개???????
# 0.95 : n_components=154
# 0.99 : n_components=331
# 0.999 : n_components=486
# 1.0 : n_components=706, 713
# np.argmax 써라
#####################################################

pca = PCA(n_components=28*28)
x_train_reshape = pca.fit_transform(x_train_reshape)
# print(x.shape)          

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum>1) + 1)

result_val = [0.95 ,0.99, 0.999, 1.0]
for i in result_val:
    print(i,np.argmax(cumsum>i) + 1)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()