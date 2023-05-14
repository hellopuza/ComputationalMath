import math as m
import numpy as np
from computations.sys_eq import tridiagonal

rd_angle     = lambda u, i, n, h, t, a: u[i, n] - t * a / h * (u[i, n] - u[i - 1, n])
ru_angle     = lambda u, i, n, h, t, a: (u[i - 1, n] * a * t + h * u[i, n - 1]) / (h + a * t)
lax_wendroff = lambda u, i, n, h, t, a: u[i, n] - 0.5 * t * a / h * (u[i + 1, n] * (1.0 - t * a / h) - u[i - 1, n] * (1.0 + t * a / h) + u[i, n] * t * a / h * 2.0)

therma_d  = lambda u, i, n, h, t, a: a * a * t / h / h * (u[i + 1, n] - 2.0 * u[i, n] + u[i - 1, n]) + u[i, n]

def init_2dim_grid_conv(gf, phi, psi):
    for n in range(gf.n[1]):
        gf.y[0, n] = psi(gf.x[0, n][1])

    for i in range(gf.n[0]):
        gf.y[i, 0] = phi(gf.x[i, 0][0])

def init_2dim_grid_therma(gf, u1, u2, u3):
    for i in range(gf.n[0]):
        gf.y[i, 0] = u3(gf.x[i, 0][0])

    for n in range(gf.n[1]):
        gf.y[0, n] = u1(gf.x[0, n][1])
        gf.y[-1, n] = u2(gf.x[-1, n][1])

def right11_ex(gf, phi, psi, a):
    init_2dim_grid_conv(gf, phi, psi)
    for n in range(1, gf.n[1]):
        for i in range(1, gf.n[0]):
            gf.y[i, n] = rd_angle(gf.y, i, n - 1, gf.h[0], gf.h[1], a)

def right11_im(gf, phi, psi, a):
    init_2dim_grid_conv(gf, phi, psi)
    for n in range(1, gf.n[1]):
        for i in range(1, gf.n[0]):
            gf.y[i, n] = ru_angle(gf.y, i, n, gf.h[0], gf.h[1], a)

def lax_wendroff22_ex(gf, phi, psi, a):
    init_2dim_grid_conv(gf, phi, psi)
    for n in range(1, gf.n[1]):
        for i in range(1, gf.n[0] - 1):
            gf.y[i, n] = lax_wendroff(gf.y, i, n - 1, gf.h[0], gf.h[1], a)
        i = gf.n[0] - 1
        gf.y[i, n] = rd_angle(gf.y, i, n - 1, gf.h[0], gf.h[1], a)

def therma21_ex(gf, u1, u2, u3, a):
    init_2dim_grid_therma(gf, u1, u2, u3)
    for n in range(1, gf.n[1]):
        for i in range(1, gf.n[0] - 1):
            gf.y[i, n] = therma_d(gf.y, i, n - 1, gf.h[0], gf.h[1], a)

def therma21_im(gf, u1, u2, u3, a):
    init_2dim_grid_therma(gf, u1, u2, u3)
    for n in range(1, gf.n[1]):
        a_ = c_ = np.asarray([-a * a * gf.h[1] / gf.h[0] / gf.h[0]] * (gf.n[0] - 2))
        b_ = np.asarray([1.0 + 2.0 * a * a * gf.h[1] / gf.h[0] / gf.h[0]] * (gf.n[0] - 2))
        d_ = np.array(gf.y[1:gf.n[0] - 1, n - 1])
        d_[0] -= c_[0] * gf.y[0, n]
        d_[-1] -= a_[0] * gf.y[gf.n[0] - 1, n]
        gf.y[1:gf.n[0] - 1, n] = tridiagonal(a_, b_, c_, d_)

def crunk_nicholson22_im(gf, u1, u2, u3, a):
    init_2dim_grid_therma(gf, u1, u2, u3)
    for n in range(1, gf.n[1]):
        a_ = c_ = np.asarray([-0.5 * a * a * gf.h[1] / gf.h[0] / gf.h[0]] * (gf.n[0] - 2))
        b_ = np.asarray([1.0 + a * a * gf.h[1] / gf.h[0] / gf.h[0]] * (gf.n[0] - 2))
        d_ = np.array(gf.y[1:gf.n[0] - 1, n - 1])
        d_ = d_ + 0.5 * a * a * gf.h[1] / gf.h[0] / gf.h[0] * (gf.y[2:gf.n[0], n - 1] + gf.y[0:gf.n[0] - 2, n - 1] - 2.0 * d_)
        d_[0] -= c_[0] * gf.y[0, n]
        d_[-1] -= a_[0] * gf.y[gf.n[0] - 1, n]
        gf.y[1:gf.n[0] - 1, n] = tridiagonal(a_, b_, c_, d_)

def cross22_im(gf, f, u1, u2, u3, u4, eps):
    for n in range(gf.n[1]):
        gf.y[0, n] = u1(gf.x[0, n][1])
        gf.y[-1, n] = u2(gf.x[-1, n][1])

    for i in range(gf.n[0]):
        gf.y[i, 0] = u3(gf.x[i, 0][0])
        gf.y[i, -1] = u4(gf.x[i, -1][0])

    max_iters = int(m.fabs(gf.n[0] * gf.n[1] / np.pi / np.pi * m.log(1.0 / eps)))
    for iter in range(max_iters):
        for i in range(1, gf.n[0] - 1):
            for j in range(1, gf.n[1] - 1):
                hx = gf.h[0]
                hy = gf.h[1]
                gf.y[i, j] = 0.5 / (hx * hx + hy * hy) * (hx * hx * (gf.y[i, j + 1] + gf.y[i, j - 1]) + hy * hy * (gf.y[i + 1, j] + gf.y[i - 1, j]) - hx * hx * hy * hy * f(gf.x[i, j]))

def euler_ex(gf, f):
    for i in range(1, gf.n):
        gf.y[i] = gf.y[i - 1] + gf.h * f(gf.y[i - 1], gf.x[i - 1])

def rk2_ex(gf, f):
    for i in range(1, gf.n):
        gf.y[i] = gf.y[i - 1] + gf.h * f(gf.y[i - 1] + gf.h / 2 * f(gf.y[i - 1], gf.x[i - 1]), gf.x[i - 1] + gf.h / 2)

def rk4_ex(gf, f):
    for i in range(1, gf.n):
        k1 = f(gf.y[i - 1])
        k2 = f(gf.y[i - 1] + gf.h * k1 * 0.5)
        k3 = f(gf.y[i - 1] + gf.h * k2 * 0.5)
        k4 = f(gf.y[i - 1] + gf.h * k3)
        gf.y[i] = gf.y[i - 1] + gf.h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def rk4_im(gf, f):
    c1 = 0.5 - m.sqrt(3.0) / 6.0
    c2 = 0.5 + m.sqrt(3.0) / 6.0
    a12 = 0.25 - m.sqrt(3.0) / 6.0
    a21 = 0.25 + m.sqrt(3.0) / 6.0
    b1 = b2 = 0.5
    b1_ = 0.5 + m.sqrt(3.0) * 0.5
    b2_ = 0.5 - m.sqrt(3.0) * 0.5
    a11 = a22 = 0.25

    f1 = lambda k1, k2: f(gf.y[i - 1] + gf.h * (a11 * k1 + a12 * k2))
    f2 = lambda k1, k2: f(gf.y[i - 1] + gf.h * (a21 * k1 + a22 * k2))

    for i in range(1, gf.n):
        k1 = f(gf.y[i - 1] + gf.h * c1 * f(gf.y[i - 1]))
        k2 = f(gf.y[i - 1] + gf.h * c1 * f(gf.y[i - 1]))

        for j in range(4):
            k1, k2 = f1(k1, k2), f2(k1, k2)

        gf.y[i] = gf.y[i - 1] + gf.h * (k1 * b1 + k2 * b2)