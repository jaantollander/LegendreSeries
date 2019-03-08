import numpy as np

from legendre_series import step_function_coefficients, step_function
from plots import plot_legendre_series


left_edge = -1.0
right_edge = 1.0
a = 0.50
x = np.linspace(left_edge, right_edge, 1001)
special_points = [-a, a, 0.0, left_edge, right_edge]
x = np.union1d(x, special_points)

n = 100
plot_legendre_series(x, a, n, step_function_coefficients,
                     f"step_function_series_{n}", step_function,
                     ylim_min=1e-6, save=True)
