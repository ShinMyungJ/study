from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()

# x = datasets.data
# y = datasets.target
# print(type(x))

irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(irisDF)

kmeans = KMeans(n_clusters=3, random_state=66)  # n_clusters 몇 개의 군집으로 나눌지

# KMeans의 param default 값
# n_clusters = 8, init='k-means++', n_init=10, max_iter=300,
# tol=0.0001, verbose=0, # random_state=None, copy_x=True, algorithm='auto'

kmeans.fit(irisDF)

print(kmeans.labels_)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
#  2 1]

print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# irisDF['cluster'] = kmeans.labels_

# irisDF['target'] = datasets.target

# iris_result = irisDF.groupby(['target', 'cluster'])['sepal_length'].count()
# print(iris_result)

print(accuracy_score(kmeans.labels_, datasets.target))

# 0.8933333333333333



# 비지도학습 : y가 없는 것!
# y값이 없을때 y를 생성해야할 때 사용할 수 있음
# 분류 모델에만 사용 가능

