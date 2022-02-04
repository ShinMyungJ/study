import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
# print(x)
# print(y)
# print(type(x))     # <class 'numpy.ndarray'>

# df = pd.DataFrame(x, columns=datasets['feature_names'])
# df = pd.DataFrame(x, columns=datasets.feature_names)
df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
print(df)

df['Target(Y)'] = y
print(df)

print("====================== 상관계수 히트 맵 ====================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()

