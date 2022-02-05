import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
print(data.shape)   # (4, 5)
data = data.transpose()
data.columns = ['a', 'b', 'c', 'd']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

imputer = SimpleImputer(strategy='mean')
imputer2 = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')
# imputer = SimpleImputer(strategy='constant', fill_value=777)

imputer.fit(data)
data2 = imputer.transform(data)
print(data2)

# fit에는 dataframe이 들어가는데, 우리는 칼럼만 바꾸고 싶다.
# 시리즈를 넣으면 에러가 난다
# 처리해 보아라!!!

data2 = data
imputer.fit(data[['a','b']])
data2[['a','b']] = imputer.transform(data[['a','b']])
print(data2)

imputer2.fit(data[['c']])
data2['c'] = imputer2.transform(data[['c']])
print(data2)