import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston, load_iris

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

datasets = load_iris()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)
print(type(x_train),type(y_train))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

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
        
model = Model(4, 3).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss()
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward() # 역전파
    optimizer.step()
    return loss.item()

EPOCHS = 1001
for epoch in range(1, EPOCHS):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch: {}, loss :{}".format(epoch, loss))

#4. 평가, 예측
print("====================평가, 예측=========================")
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print('최종loss : ', loss)

y_predict = torch.argmax(model(x_test), 1)
print(y_predict[:10])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu().detach().numpy(), 
                       y_predict.cpu().detach().numpy())
print('accuracy : {:.4f}'.format(score))

# ================ 평가, 예측 ================
# 최종loss :  0.9216799139976501
# tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 2], device='cuda:0')
# accuracy : 0.9333
# accuracy : 0.9333

