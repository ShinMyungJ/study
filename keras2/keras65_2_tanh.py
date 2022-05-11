import numpy as np
import matplotlib.pyplot as plt

def tanh(x, diff=False):
    if diff:
        return (1+tanh(x))*(1-tanh(x))
    else:
        return np.tanh(x)
x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()
