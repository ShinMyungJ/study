import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터                   # 데이터 정제과정, 중요!
x = np.array([range(10), range(21,31), range(201, 211)])
print(x)
x = np.transpose(x)
print(x.shape)              # (10, 3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.5, 1.4, 1.3],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.transpose(y)
print(y.shape)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

print(x, y)
print(x.shape, y.shape)

#2. 모델
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.Linear(16, 32),
    nn.Linear(32, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 3),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()       # 훈련모드
    optimizer.zero_grad()   # 기울기 초기화
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss(hypothesis, y)  # error, 받아들이는 객체가 없음
    # loss = nn.MSELoss()(hypothesis, y)
    # loss = F.mse_loss(hypothesis, y)
    
    loss.backward()     # 기울기 값 계산까지
    optimizer.step()    # 가중치 수정
    return loss.item()
    
epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch, loss))
    
print("=================================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()        # 평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
        return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

result = model(torch.Tensor([[9, 30, 210]]).to(DEVICE))
print('[9, 30, 210]의 예측값 : ', result)

