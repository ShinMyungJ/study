from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150,4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ",len(allAlgorithms))   # 모델의 갯수 :  41(classifier) / 54(regressor)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률 : ', scores)
    except:
        # continue
        print(name, '은 에러 터진 놈!!!!')
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# allAlgorithms :  [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', <class 'sklearn.linear_model._logistic.LogisticRegression'>), ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>), ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]
# 모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  [0.93406593 0.96703297 0.95604396 1.         0.96703297]
# BaggingClassifier 의 정답률 :  [0.9010989  0.91208791 0.97802198 0.96703297 0.97802198]
# BernoulliNB 의 정답률 :  [0.59340659 0.56043956 0.59340659 0.63736264 0.68131868]
# CalibratedClassifierCV 의 정답률 :  [0.92307692 0.96703297 0.97802198 1.         0.96703297]
# CategoricalNB 은 에러 터진 놈!!!!
# ClassifierChain 은 에러 터진 놈!!!!
# ComplementNB 의 정답률 :  [0.8021978  0.83516484 0.8021978  0.85714286 0.86813187]
# DecisionTreeClassifier 의 정답률 :  [0.93406593 0.85714286 0.96703297 0.91208791 0.93406593]
# DummyClassifier 의 정답률 :  [0.58241758 0.57142857 0.61538462 0.65934066 0.71428571]
# ExtraTreeClassifier 의 정답률 :  [0.83516484 0.89010989 0.97802198 0.94505495 0.92307692]
# ExtraTreesClassifier 의 정답률 :  [0.91208791 0.95604396 0.97802198 0.98901099 0.97802198]
# GaussianNB 의 정답률 :  [0.84615385 0.91208791 0.97802198 0.96703297 0.94505495]
# GaussianProcessClassifier 의 정답률 :  [0.9010989  0.94505495 0.96703297 0.98901099 0.97802198]
# GradientBoostingClassifier 의 정답률 :  [0.91208791 0.91208791 0.97802198 0.96703297 0.97802198]
# HistGradientBoostingClassifier 의 정답률 :  [0.93406593 0.94505495 0.97802198 0.97802198 0.97802198]
# KNeighborsClassifier 의 정답률 :  [0.92307692 0.94505495 0.98901099 1.         0.94505495]
# LabelPropagation 의 정답률 :  [0.92307692 0.93406593 0.97802198 1.         0.97802198]
# LabelSpreading 의 정답률 :  [0.91208791 0.93406593 0.96703297 0.98901099 0.97802198]
# LinearDiscriminantAnalysis 의 정답률 :  [0.91208791 0.93406593 0.95604396 0.97802198 0.97802198]
# LinearSVC 의 정답률 :  [0.94505495 0.97802198 0.97802198 1.         0.96703297]
# LogisticRegression 의 정답률 :  [0.9010989  0.94505495 0.96703297 1.         0.97802198]
# LogisticRegressionCV 의 정답률 :  [0.92307692 0.97802198 0.96703297 1.         0.96703297]
# MLPClassifier 의 정답률 :  [0.92307692 0.98901099 0.97802198 0.98901099 0.96703297]
# MultiOutputClassifier 은 에러 터진 놈!!!!
# MultinomialNB 의 정답률 :  [0.79120879 0.76923077 0.85714286 0.85714286 0.89010989]
# NearestCentroid 의 정답률 :  [0.85714286 0.89010989 0.96703297 0.95604396 0.95604396]
# NuSVC 의 정답률 :  [0.87912088 0.92307692 0.96703297 0.96703297 0.96703297]
# OneVsOneClassifier 은 에러 터진 놈!!!!
# OneVsRestClassifier 은 에러 터진 놈!!!!
# OutputCodeClassifier 은 에러 터진 놈!!!!
# PassiveAggressiveClassifier 의 정답률 :  [0.92307692 0.98901099 0.98901099 0.98901099 0.95604396]
# Perceptron 의 정답률 :  [0.92307692 1.         0.95604396 0.93406593 0.89010989]
# QuadraticDiscriminantAnalysis 의 정답률 :  [0.92307692 0.94505495 0.97802198 1.         0.91208791]
# RadiusNeighborsClassifier 은 에러 터진 놈!!!!
# RandomForestClassifier 의 정답률 :  [0.91208791 0.96703297 0.96703297 0.95604396 0.97802198]
# RidgeClassifier 의 정답률 :  [0.89010989 0.93406593 0.96703297 0.98901099 0.97802198]
# RidgeClassifierCV 의 정답률 :  [0.91208791 0.93406593 0.97802198 0.98901099 0.98901099]
# SGDClassifier 의 정답률 :  [0.96703297 0.91208791 0.94505495 0.98901099 0.96703297]
# SVC 의 정답률 :  [0.94505495 0.95604396 0.97802198 1.         0.96703297]
# StackingClassifier 은 에러 터진 놈!!!!
# VotingClassifier 은 에러 터진 놈!!!!