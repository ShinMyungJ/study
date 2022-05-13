import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# np에서 tensor형태로 변환

x = torch.FloatTensor(x).unsqueeze(1)   # (3,) -> (3, 1)
y = torch.FloatTensor(y).unsqueeze(1)   # (3,) -> (3, 1)

print(x, y) # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape, y.shape) # torch.Size([3]) torch.Size([3])

# (3, ) -> (3, 1)형태로 reshape 해줘야함

#2. 모델
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1)     # 인풋, 아웃풋


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# print(optimizer)

# model.fit(x, y, epochs=1000, batch_size=1)
def train(model, criterion, optimizer, x, y):
    # model.train()       # 훈련모드
    optimizer.zero_grad()   # 기울기 초기화
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()     # 기울기 값 계산까지
    optimizer.step()    # 가중치 수정
    return loss.item()
    
epochs = 500
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch, loss))
    
print("=================================================")

#4. 평가, 예측
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y):
    model.eval()        # 평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# result = model.predict([4])
result = model(torch.Tensor([[4]]))
print('4의 예측값 : ', result.item())

# 최종 loss :  6.02208283240202e-09
# 4의 예측값 :  4.000161647796631
