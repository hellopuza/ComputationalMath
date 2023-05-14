import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

class GridFunction:
    def __init__(self, a, b, h, ydim=1):
        a = np.asarray(a)
        b = np.asarray(b)
        h = np.asarray(h)

        self.__shape = lambda obj: (1,) if np.shape(obj) == () else np.shape(obj)
        self.__len = lambda obj: self.__shape(obj)[0]

        a_shape = self.__shape(a)
        b_shape = self.__shape(b)
        h_shape = self.__shape(h)

        if a_shape != b_shape:
            raise ValueError(f"Shapes of a '{a_shape}' and b '{b_shape}' must be the same")
        elif a_shape != h_shape:
            raise ValueError(f"Shapes of a '{a_shape}' and h '{h_shape}' must be the same")

        self.ydim = ydim
        self.xdim = self.__len(a)

        if self.xdim > 1:
            self.n = ((b - a) / h).astype(int) + np.ones(self.xdim, dtype=int)
        else:
            self.n = int((b - a) / h) + 1

        self.a = a
        self.b = a + h * (self.n - 1)
        self.h = h

        if self.xdim == 1:
            self.x = np.zeros(self.n)
            for i in range(self.n):
                self.x[i] = a + self.h * i
        else:
            self.x = np.zeros(np.append(self.n, self.xdim))
            for i in np.ndindex(tuple(self.n)):
                self.x[i] = a + self.h * i

        if self.ydim == 0:
            self.y = [[0]] * self.n
        elif self.ydim == 1:
            self.y = np.zeros(self.n)
        else:
            self.y = np.zeros(np.append(self.n, self.ydim))

    def calc(self, func):
        self.f = func
        if self.xdim == 1:
            for i in range(self.n):
                self.y[i] = func(self.x[i])
        else:
            for i in np.ndindex(tuple(self.n)):
                self.y[i] = func(self.x[i])

    def plot(self, xaxis=['x', 0], yaxis=['y', [0], 0], zaxis=None, marker='', label='', title_str='', figsize=(9, 6), new=True, ext_plt=None):
        if new and ext_plt is None and zaxis is None:
            fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set(xlabel='x', ylabel='y', title=title_str)
            self.ax.grid()

        if ext_plt is not None:
            self.ax = ext_plt

        def get_axis(axis):
            if axis[0] == 'x':
                if self.xdim == 1:
                    return self.x

                axis = axis[1]
                slices = [slice(None) if i == axis else axis if i == self.xdim else 0
                          for i in range(self.xdim + 1)]
                return self.x[slices]

            elif axis[0] == 'y':
                xax = axis[1]
                yax = axis[2]
                if self.__len(xax) != self.xdim:
                    raise ValueError(f"Dimension of axis[1] '{xax}' must be '{self.xdim}'")

                if self.xdim == 1:
                    if self.ydim == 1:
                        return self.y

                    return self.y[:, yax]

                if self.ydim == 1:
                    slices = [slice(None) if i == xax[0] else xax[i] if i > xax[0] else xax[i + 1]
                              for i in range(self.xdim)]
                    return self.y[slices]

                slices = [slice(None) if i == xax[0] else yax if i == self.xdim else xax[i] if i > xax[0] else xax[i + 1]
                          for i in range(self.xdim + 1)]
                return self.y[slices]

            elif axis[0] == 'z':
                xax = axis[1]
                yax = axis[2]

                if self.ydim == 1:
                    slices = [slice(None) if i == xax[0] else xax[1]
                              for i in range(self.xdim - 1)]
                    return self.y[slices]

                slices = [slice(None) if i == xax[0] else yax if i == self.xdim else xax[1]
                          for i in range(self.xdim)]
                return self.y[slices]

            else:
                raise ValueError(f"axis[0] '{axis[0]}' must be 'x' or 'y' or 'z")

        x = get_axis(xaxis)
        y = get_axis(yaxis)
        z = None if zaxis is None else get_axis(zaxis)

        if z is None:
            if self.ydim == 0:
                if label == '':
                    self.ax.scatter(x, y, s=0.3)
                else:
                    self.ax.scatter(x, y, s=0.3, label=label)
                    self.ax.legend()
            else:
                if label == '':
                    self.ax.plot(x, y, marker)
                else:
                    self.ax.plot(x, y, marker, label=label)
                    self.ax.legend()
        else:
            x, y = np.meshgrid(x, y)
            x = np.transpose(x)
            y = np.transpose(y)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set(title=title_str)
            ax.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=cm.coolwarm)