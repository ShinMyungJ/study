from sklearn.utils import all_estimators
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = fetch_covtype()

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
        print("scores : ", scores, "\n cross_val_score", round(np.mean(scores),4))

    except:
        # continue
        print(name, '은 에러 터진 놈!!!!')
    
# 오래된 algorithm or version 으로 인한 문제로 결과값이 안나오는 경우가 있음

# 모델의 갯수 :  41
# AdaBoostClassifier 의 정답률 :  [0.42442073 0.45958564 0.44924808 0.53531551 0.56334377]
#  cross_val_score 0.4864
# BaggingClassifier 의 정답률 :  [0.9581657  0.95762785 0.95749876 0.95654138 0.95702499]
#  cross_val_score 0.9574
# BernoulliNB 의 정답률 :  [0.63046191 0.63131172 0.63288225 0.63283922 0.63394327]
#  cross_val_score 0.6323
# CalibratedClassifierCV 의 정답률 :  [0.71226953 0.71254921 0.71257073 0.71301177 0.71320231]
#  cross_val_score 0.7127
# CategoricalNB 의 정답률 :  [       nan        nan 0.63317269        nan        nan]
#  cross_val_score nan
# ClassifierChain 은 에러 터진 놈!!!!
# ComplementNB 의 정답률 :  [0.61822035 0.62128612 0.61986618 0.61832792 0.61893697]
#  cross_val_score 0.6193
# DecisionTreeClassifier 의 정답률 :  [0.93419892 0.93127299 0.93268217 0.93125148 0.93217586]
#  cross_val_score 0.9323
# DummyClassifier 의 정답률 :  [0.48745724 0.48802737 0.4889955  0.48923216 0.48601026]
#  cross_val_score 0.4879
# ExtraTreeClassifier 의 정답률 :  [0.85060562 0.84092425 0.83640627 0.8567264  0.86598681]
#  cross_val_score 0.8501
# GaussianNB 의 정답률 :  [0.09272606 0.09190852 0.08909017 0.09090811 0.08831661]
#  cross_val_score 0.0906