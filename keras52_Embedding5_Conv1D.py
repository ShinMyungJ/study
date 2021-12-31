from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밋어요', '참 최고예요', '참 잘 만든 영화예요',
       '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요',
       '별로예요', '생각보다 지루해요', '연기가 어색해요',
       '재미없어요', '너무 재미없다', '참 재밋네요', '예람이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)


# {'참': 1, '너무': 2, '잘': 3, '재밋어요': 4, '최고예요': 5, '만든': 6, '영화예요': 7, '추천하고': 8,
# '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16,
# '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22,
# '재미없다': 23, '재밋네요': 24, '예람이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19],
#  [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x)
print(pad_x.shape)      # (13, 5)

word_size = len(token.word_index)
print("word_size : ", word_size)        # word_size :  27

print(np.unique(pad_x)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                        # 24 25 26 27]

# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]

# 음악, 말하기 전부 시계열 데이터

# 원핫 인코딩하면 뭐로 바뀌어? (13, 5) -> (13, 5, 28)
# 옥스포드 사전은? (13, 5, 1000000) -> 6500만개 : 요러면 망!!

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

#2. 모델
model = Sequential()
#                                               인풋은 (13, 5)
#                  단어사전의 갯수                 단어수, 길이         # 벡터화 시키는 것
# model.add(Embedding(input_dim=30, output_dim=10, input_length=5))   # input_dim : 단어사전의 갯수보다 적게하면 안되는 경우 있음
# model.add(Embedding(28, 20, input_length=5))       # embedding에서만 input_dim : 단어사전의 갯수(종류)
model.add(Embedding(50, 10, input_length=5))
model.add(Conv1D(32, 2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)

# 실습 ############################소스를 완성하거라!!!#######################
x_predict = '나는 반장이 정말 재미없다 정말'
x_predict = [x_predict]
token.fit_on_texts(x_predict)
predict_ped = token.texts_to_sequences(x_predict)
pad_predict = pad_sequences(predict_ped, padding='pre', maxlen=5)
predict = model.predict(pad_predict)

if predict >= 0.5:
    print('긍정 확률 : ', predict*100,"%")
elif predict < 0.5:
    print('부정 확률 : ', (1-predict)*100, "%")

