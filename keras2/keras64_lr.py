
##################### 마음껏 튜닝 ㄱㄱ

x = 10.0             # 임의로 바꿔도 되.
y = 10.0             # 목표값
w = 0.5             # 가중치 초기값
lr = 0.1
epochs = 400

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2       # mse
    
    print("Loss : ", round(loss, 4), "\tPredict : ", round(predict, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
