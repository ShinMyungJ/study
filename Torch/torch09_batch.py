import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

datasets = load_boston()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)
print(type(x_train),type(y_train))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)     # x, y를 합침. 다양한 방법이 있음.
test_set = TensorDataset(x_test, y_test)

print(len(train_set))       # 354
print(type(train_set))      # <class 'torch.utils.data.dataset.TensorDataset'>
print(train_set[0])         # x data의 0번째, y data의 0번째
# (tensor([-0.3550,  0.3810, -1.0714, -0.2815,  0.7585,  1.7561,  0.7023, -0.7661,
#         -0.5227, -0.8624, -2.4992,  0.3317, -0.7465], device='cuda:0'), tensor([43.1000], device='cuda:0'))

train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
test_loader = DataLoader(test_set, batch_size=36, shuffle=False)

#2. 모델
# model = nn.Sequential(
#     nn.Linear(30, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 8),
#     nn.ReLU(),
#     nn.Linear(8, 4),
#     nn.ReLU(),
#     nn.Linear(4, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)       
        out = self.linear4(x)
        
        return out
        
model = Model(13, 1).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()     # 역전파
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
    
epoch = 0
early_stopping = 0
best_loss = 10000
while True:
    epoch += 1
    loss = train(model, criterion, optimizer, train_loader)
    
    print(f'epoch: {epoch}, loss:{loss:.8f}')
        
    if loss < best_loss: 
        best_loss = loss    
        early_stopping = 0
    else:
        early_stopping += 1
    
    if early_stopping == 100: break

#4. 평가, 예측
print("================ 평가, 예측 ================")    
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
    
    return total_loss

loss = evaluate(model, criterion, test_loader)  
print('loss : {:.4f}'.format(loss))  

# y_predict = (model(x_test) >= 0.5).float()
# print(y_predict[:10])
y_predict = model(x_test)

# score = (y_predict == y_test).float().mean()
# print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
# score = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())
# print('accuracy : {:.4f}'.format(score))
score = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print('R2 score : {:.4f}'.format(score))

# ================ 평가, 예측 ================
# loss : 52.3388
# R2 score : 0.8743