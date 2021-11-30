import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1. 데이터

path = "./_data/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0) # header : 0 디폴트값 헤더 위치를 설정할 수 있음. header가 없으면 None
                                                               # index_col : None 디폴트값. 첫 칼럼이 인덱스인 경우 0으로해서 날릴 수 있음
print(train)
print(train.shape)              # (891, 12)    read_csv 헤더 조절, 인덱스 조절
y_train = train.Survived
print(y_train)                  # train 에서 survived 카테고리만 빼냄 
print(train.shape)


test = pd.read_csv(path + "test.csv", index_col=0, header=0)
gender_submission = pd.read_csv(path + "gender_submission.csv", index_col=0, header=0)
print(test.shape)               # (418, 11)
print(gender_submission.shape)  # (418, 2)
# print(gender_submission)

# print(train.info())
print(train.describe())


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# #2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(5, activation='linear', input_dim=30))   # 히든레이어에 sigmoid를 중간중간 사용해도 된다
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])        
#                     # 정확도 'accuracy' 결과값에 accuracy 출력됨. 훈련에 영향을 끼치지 않고 상황만 보여줌
#                     # metrics : 2개 이상 list. accuracy 외에 추가적으로 사용할 수 있음.
# model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # callbacks : 2개 이상 list

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)    # 결과값 loss : [xxxxxxx, xxxxxxx]  처음값은 loss, 두번째값은 accuracy <- 보조지표 값이 한쪽으로 치우쳐져 있으면
# print('loss : ', loss)                                                                    #                      지표로서 가치가 떨어짐
# results = model.predict(x_test)