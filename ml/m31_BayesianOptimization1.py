from bayes_opt import BayesianOptimization

def black_box_function(x, y):
    return -x **2 - (y - 1) ** 2 + 1

pbounds = {'x' : (2, 4), 'y' : (-3, 3)}

optimizer = BayesianOptimization(
    f = black_box_function,         # f 에는 model
    pbounds=pbounds,                # pbound 에는 parameter를 넣으면 됨
    random_state=66
)

optimizer.maximize(
    init_points=2,
    n_iter=15
)

# 인공지능이 아니고, 알고리즘임
# 상당히 잘 맞는편
# 가우시안 최적화 정리해라

