import numpy as np

from plots import plot_piecewise_functions

a = 0.5
x = np.union1d(np.linspace(-1, 1, 1001), [a])
plot_piecewise_functions(x, a)
