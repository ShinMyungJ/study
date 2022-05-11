import numpy as np
import matplotlib.pyplot as plt
from tkinter import Y

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y
plt.pie(y, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# softmax의 전체의 합은 1이다. 
# activation의 주목적은 '한정'하는 것이다. 
# 다음 레이어로 전달 할 때 제한한다.  >> 제한하지 않을 시에 생기는 문제들을 해결하기 위해 
