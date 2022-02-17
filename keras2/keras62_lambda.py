
gradient = lambda x: 2*x - 4        # :뒤는 return 값, gradient는 함수명!
                                    # 함수명 = lambda x: 와 def 함수명(x)는 같음

def gradient2(x):
    return (2*x - 4 )

# def gradient3(x):
#     temp = 2*x - 4
#     return temp

# gradient 와 gradient2는 같음

x = 3

print(gradient(x))
print(gradient2(x))
