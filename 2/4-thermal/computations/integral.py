def riemann_left(gf, step = 1):
    sum = 0
    for i in range(0, gf.n - 1, step):
        sum += gf.y[i]
    return sum * gf.h * step

def riemann_right(gf):
    sum = 0
    for i in range(1, gf.n):
        sum += gf.y[i]
    return sum * gf.h

def riemann_central(gf):
    sum = (gf.y[0] + gf.y[-1]) / 2
    for i in range(1, gf.n - 1):
        sum += gf.y[i]
    return sum * gf.h

def trapezoidal(gf, step = 1):
    sum = 0
    for i in range(0, gf.n - 1, step):
        sum += gf.y[i] + gf.y[i + step]
    return sum * gf.h * step / 2

def simpson(gf):
    sum = 0
    for i in range(1, gf.n - 1, 2):
        sum += gf.y[i - 1] + 4 * gf.y[i] + gf.y[i + 1]
    return sum * gf.h / 3