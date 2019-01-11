import numpy as np

from legendre_series import step_function_coefficients, step_function, \
    v_function_coefficients, v_function
from plots import animate_legendre_series, plot_pointwise_convergence


x = np.linspace(-1, 1, 1001)
a = 0.5  # TODO: 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999
n = 1000  # frames
near = [m*10**(-n) for n in range(1, 8) for m in range(1, 10)]


animate_legendre_series(x, a, n,
                        step_function_coefficients,
                        f"step_function_series_{n}",
                        step_function)

animate_legendre_series(x, a, n,
                        v_function_coefficients,
                        f"v_function_series_{n}",
                        v_function)


plot_pointwise_convergence(1-1e-7, 0.5, int(1e5),
                           step_function_coefficients,
                           "step_function",
                           step_function,
                           0)

plot_pointwise_convergence(1-1e-7, 0.5, int(1e5),
                           v_function_coefficients,
                           "v_function",
                           v_function,
                           1)
