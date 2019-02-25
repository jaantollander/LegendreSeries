import numpy as np

from legendre_series import step_function_coefficients, step_function, \
    v_function_coefficients, v_function
from plots import animate_legendre_series


left_edge = -1.0
right_edge = 1.0
a = 0.50
x = np.linspace(left_edge, right_edge, 1001)
special_points = [-a, a, 0.0, left_edge, right_edge]
x = np.union1d(x, special_points)


frames = 500
animate_legendre_series(x, a, frames,
                        step_function_coefficients,
                        f"step_function_series_{frames}",
                        step_function,
                        save=True)

frames = 500
animate_legendre_series(x, a, frames,
                        v_function_coefficients,
                        f"v_function_series_{frames}",
                        v_function,
                        save=True)
