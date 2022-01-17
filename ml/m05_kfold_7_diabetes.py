import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = RandomForestRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("r2 : ", scores, "\n cross_val_score", round(np.mean(scores),4))

# r2 :  [0.4793128  0.54120904 0.39253717 0.54934236 0.44223808] 
#  cross_val_score 0.4809