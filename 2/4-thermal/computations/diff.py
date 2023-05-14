import math as m
import numpy as np

def diff_coef(d, l, r): # f'(x_i) = sum k from -l to r (c_lr / h * f(x_i + k * h))
    n = r + l + 1
    A = np.zeros((n, n))
    f = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            A[i][j] = (-l + j) ** i

    f[d] = 1 / m.factorial(d)
    return np.linalg.solve(A, f)

def calc_diff(f, x, h, d, n):
    coefs = diff_coef(d, 0, n - 1)
    sum = 0.0
    for i in range(n):
        sum += f(x + h * i) * coefs[i]
    return sum / m.pow(h, d)
