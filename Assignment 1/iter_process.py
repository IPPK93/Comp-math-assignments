import numpy as np

def iter_process(func, x_0, iter_num):
    x = x_0
    for i in range(iter_num):
        x = func(x)
    return x

def epsilon_iter_process(func, x_0, epsilon):
    x_prev = x_0
    x_curr = func(x_0)
    
    iter_num = 1
    while np.abs(x_curr - x_prev) >= epsilon:
        x_prev = x_curr
        x_curr = func(x_curr)
        iter_num += 1
    return x_curr, iter_num