import numpy as np
aaa = np.array([1,2,-20, 4, 5, 6, 7, 8, 30])#, 100, 500])#, 12, 13])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | : or , 조건에 맞는 값의 위치값

outlier_loc = outliers(aaa)
print("이상치의 위치 : ", outlier_loc)

# 시각화

import matplotlib.pyplot as plt
import seaborn as sns
# plt.boxplot(aaa, sym='bo')
plt.boxplot(aaa, notch=1, sym='b*', vert=0)
# sns.boxplot(data = aaa)
plt.title('Box plot of aaa')
plt.show()