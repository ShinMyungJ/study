import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


per_clf = Perceptron()
ls_clf = LinearSVC()
svc_clf = SVC()
knn_clf = KNeighborsClassifier()
log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
rnd_clf = RandomForestClassifier()

#3. 컴파일, 훈련

voting_clf = VotingClassifier(estimators=[('per',per_clf),
                                          ('ls',ls_clf),
                                          ('svc',svc_clf),
                                          ('knn',knn_clf),
                                         ('log',log_clf),
                                         ('tr',tree_clf),
                                         ('rn',rnd_clf)],
                             voting='hard')

voting_clf.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score

for clf in (per_clf, ls_clf, svc_clf, knn_clf, log_clf, tree_clf, rnd_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))

# Perceptron 0.8
# LinearSVC 1.0
# SVC 1.0
# KNeighborsClassifier 1.0
# LogisticRegression 1.0
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# VotingClassifier 1.0