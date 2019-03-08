import numpy as np

from plots import plot_legendre_polynomials

x = np.linspace(-1, 1, 1001)
plot_legendre_polynomials(x, n=5)
