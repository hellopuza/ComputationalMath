import numpy as np

def tridiagonal(a, b, c, d):
    nf = len(d)
    for it in range(1, nf):
        mc = a[it] / b[it - 1]
        b[it] = b[it] - mc * c[it - 1]
        d[it] = d[it] - mc * d[it - 1]

    x = b
    x[-1] = d[-1] / b[-1]
    for il in range(nf - 2, -1, -1):
        x[il] = (d[il] - c[il] * x[il + 1]) / b[il]

    return x

def gaussian(A, f):
    A = A.copy()
    f = f.copy()
    x = np.zeros((len(A), 1))

    for row in range(len(A) - 1):
        ind = np.argmax(A[row:, row]) + row
        A[row], A[ind] = A[ind].copy(), A[row].copy()
        f[row], f[ind] = f[ind].copy(), f[row].copy()

        for i in range(row + 1, len(A)):
            a = A[i][row]
            A[i] -= A[row] * a / A[row][row]
            f[i] -= f[row] * a / A[row][row]

    for row in range(len(A) - 1, -1, -1):
        sum = 0
        for i in range(row + 1, len(A)):
            sum -= x[i] * A[row][i]

        sum += f[row]
        x[row] = sum / A[row][row]

    return x

def jacobi(A, f, eps):
    x = np.zeros((len(A), 1))
    nx = np.ones((len(A), 1))
    iter_num = 0

    while True:
        x = nx.copy()

        for i in range(len(A)):
            sum = 0
            for j in range(len(A)):
                sum += A[i][j] * x[j] * (i != j)

            nx[i] = (f[i] - sum) / A[i][i]

        iter_num += 1
        if max(np.absolute(x - nx)) < eps:
            break

    return x, iter_num

def seidel(A, f, eps):
    x = np.zeros((len(A), 1))
    iter_num = 0

    while True:
        old_x = x.copy()

        for i in range(len(A)):
            sum = 0
            for j in range(len(A)):
                sum += A[i][j] * x[j] * (i != j)

            x[i] = (f[i] - sum) / A[i][i]

        iter_num += 1
        if max(np.absolute(x - old_x)) < eps:
            break

    return x, iter_num

def steepest_descent(A, f, eps):
    f = np.dot(A.transpose(), f)
    A = np.dot(A.transpose(), A)
    x = np.zeros((len(A), 1))
    iter_num = 0

    while True:
        old_x = x.copy()

        r = np.dot(A, x) - f
        tau = np.dot(r.transpose(), r) / np.dot(np.dot(A, r).transpose(), r)
        x = x - tau * r

        iter_num += 1
        if max(np.absolute(x - old_x)) < eps:
            break

    return x, iter_num

def min_discrepancy(A, f, eps):
    f = np.dot(A.transpose(), f)
    A = np.dot(A.transpose(), A)
    x = np.zeros((len(A), 1))
    iter_num = 0

    while True:
        old_x = x.copy()

        r = np.dot(A, x) - f
        Ar = np.dot(A, r)
        tau = np.dot(Ar.transpose(), r) / np.dot(Ar.transpose(), Ar)
        x = x - tau * r

        iter_num += 1
        if max(np.absolute(x - old_x)) < eps:
            break

    return x, iter_num