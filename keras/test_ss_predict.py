# Open: Opening stock price of the day
# Close: Closing stock price of the day
# High: Highest stock price of the data
# Low: Lowest stock price of the day

# 데이터 시각화
# 최고가와 최저가의 중간 값을 도표에 그려보자. 코로나 이후에 주가가 급격하게 하락한 것을 볼 수 있다.
# plt.figure(figsize = (18,9))
# df = stockPrice.copy()
# plt.plot(df.index,(df['Low']+df['High'])/2.0)
# plt.xticks(df.iloc[::50,:].index,rotation=45)
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()

# 데이터 분리
# stockPriceClose=stockPrice[['Close']]
# split_date = pd.Timestamp('01-01-2019')
# #학습용 데이터와 테스트용 데이터로 분리
# train_data=pd.DataFrame(stockPriceClose.loc[:split_date,['Close']])
# test_data=pd.DataFrame(stockPriceClose.loc[split_date:,['Close']])
# #분리된 데이터 시각화
# ax = train_data.plot()
# test_data.plot(ax=ax)
# plt.legend(['train', 'test'])

# 데이터 정규화(스케일링)
# 일반적인 정규화 순서
# Step 1: 학습 데이터로 scaler를 훈련시킨다.
# Step 2: scaler를 사용해서 학습데이터를 정규화한다.
# Step 3: 예측 모델을 학습하기 위해서 정규화된 훈련데이터를 사용한다.
# Step 4: 위에서 훈련된  scaler를 사용해서 테스트 데이터를 변형(정규화)한다.
# Step 5: 변형된 테스트 데이터와 학습된 모델을 사용해서 예측한다.

# from sklearn.preprocessing import MinMaxScaler 
# scaler = MinMaxScaler() 
# train_data_sc=scaler.fit_transform(train_data)
# test_data_sc= scaler.transform(test_data)

# 데이터 윈도우 생성
# #학습 데이터와 테스트 데이터( ndarray)를 데이터프레임으로 변형한다.
# train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
# test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)
# #LSTM은 과거의 데이터를 기반으로 미래을 예측하는 모델이다. 따라서, 과거 데이터를 몇 개 사용해서 예측할 지 정해야 한다. 여기서는 30개(한 달)를 사용한다.  
# for i in range(1, 31):
#     train_sc_df ['Scaled_{}'.format(i)]=train_sc_df ['Scaled'].shift(i)
#     test_sc_df ['Scaled_{}'.format(i)]=test_sc_df ['Scaled'].shift(i)

# #nan 값이 있는 로우를 삭제하고 X값과 Y값을 생성한다.
# x_train=train_sc_df.dropna().drop('Scaled', axis=1)
# y_train=train_sc_df.dropna()[['Scaled']]

# x_test=test_sc_df.dropna().drop('Scaled', axis=1)
# y_test=test_sc_df.dropna()[['Scaled']]

#대부분의 기계학습 모델은 데이터프레임 대신 ndarray구조를 입력 값으로 사용한다.
#ndarray로 변환한다.

# x_train=x_train.values
# x_test=x_test.values

# y_train=y_train.values
# y_test=y_test.values

#LSTM 모델에 맞게 데이터 셋 변형
# x_train_t = x_train.reshape(x_train.shape[0], 30,1)
# x_test_t = x_test.reshape(x_test.shape[0], 30, 1)

# from keras.layers import LSTM 
# from keras.models import Sequential 
# from keras.layers import Dense 
# import keras.backend as K 
# from keras.callbacks import EarlyStopping 

# K.clear_session() 
# # Sequeatial Model
# model = Sequential() 
# # 첫번째 LSTM 레이어
# model.add(LSTM(30,return_sequences=True, input_shape=(30, 1))) 
# # 두번째 LSTM 레이어
# model.add(LSTM(42,return_sequences=False))  
# # 예측값 1개
# model.add(Dense(1, activation='linear')) 
# # 손실함수 지정 - 예측 값과 실제 값의 차이를 계산한다. MSE가 사용된다. 
# # 최적화기 지정 - 일반적으로 adam을 사용한다.
# model.compile(loss='mean_squared_error', optimizer='adam') 
# model.summary()

#손실 값(loss)를 모니터링해서 성능이 더이상 좋아지지 않으면 epoch를 중단한다.
#vervose=1은 화면에 출력
# early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

# #epochs는 훈련 반복 횟수를 지정하고 batch_size는 한 번 훈련할 때 입력되는 데이터 크기를 지정한다.
# model.fit(x_train_t, y_train, epochs=50,
#           batch_size=20, verbose=1, callbacks=[early_stop])

# y_pred = model.predict(x_test_t)
# 모델 검증
# 테스트의 Y값(실측값) 과 예측값을 비교한다.
# t_df=test_sc_df.dropna()
# y_test_df=pd.DataFrame(y_test, columns=['close'], index=t_df.index)
# y_pred_df=pd.DataFrame(y_pred, columns=['close'], index=t_df.index)

# ax1=y_test_df.plot()
# y_pred_df.plot(ax=ax1)
# plt.legend(['test','pred'])