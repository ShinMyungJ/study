import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()