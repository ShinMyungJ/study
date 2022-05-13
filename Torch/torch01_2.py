# gpu
# criterion 이슈

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)
# torch :  1.10.2 사용DEVICE :  cuda
# data와 model에 gpu 명시 해야함

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# np에서 tensor형태로 변환
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   # (3,) -> (3, 1)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (3,) -> (3, 1)

print(x, y) # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape, y.shape) # torch.Size([3]) torch.Size([3])

# (3, ) -> (3, 1)형태로 reshape 해줘야함

#2. 모델
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1).to(DEVICE)     # 인풋, 아웃풋


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
    
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss(hypothesis, y)  # error, 받아들이는 객체가 없음
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward()     # 기울기 값 계산까지
    optimizer.step()    # 가중치 수정
    return loss.item()
    
epochs = 1000
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
result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 : ', result.item())

# 최종 loss :  2.929084530478576e-09
# 4의 예측값 :  3.999945878982544
