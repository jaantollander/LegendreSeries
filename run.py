import numpy as np

from legendre_series import step_function_coefficients, step_function, \
    v_function_coefficients, v_function
from plots import animate_legendre_series, plot_pointwise_convergence, \
    plot_intercepts, plot_legendre_polynomials, plot_piecewise_functions, \
    plot_intercepts_loglog

# TODO: a =  0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999
# x should include a, -1, and 1
left_edge = -1.0
right_edge = 1.0
x = np.linspace(left_edge, right_edge, 1001)
a = 0.5  # Singularity
x = np.union1d(x, [left_edge, a, right_edge])
n = 10**5
xi = np.logspace(-4, -2, 25, base=10)
near_left_a = a - xi
near_right_a = a + xi
near_left_edge = left_edge + xi
near_right_edge = right_edge - xi
special_points = [-a, a, 0.0, left_edge, right_edge]

# Add the extra points near singularity (a) and edges (-1 and 1).
x2 = np.union1d(
    np.union1d(near_left_a, near_right_a),
    np.union1d(near_left_edge, near_right_edge)
)
x2 = np.union1d(x, x2)

# Number of frames
frames = 100

# plot_legendre_polynomials(np.linspace(-1, 1, 1001))
# plot_piecewise_functions(np.linspace(-1, 1, 1001), 0.5)

# animate_legendre_series(x2, a, frames,
#                         step_function_coefficients,
#                         f"step_function_series_{frames}",
#                         step_function,
#                         save=True)
#
# animate_legendre_series(x2, a, frames,
#                         v_function_coefficients,
#                         f"v_function_series_{frames}",
#                         v_function,
#                         save=True)

# for x in special_points:
#     plot_pointwise_convergence(x, a, n,
#                                step_function_coefficients,
#                                "step_function",
#                                step_function,
#                                0,
#                                save=True)

for x in special_points:
    plot_pointwise_convergence(x, a, n,
                               v_function_coefficients,
                               "v_function",
                               v_function,
                               1,
                               save=True)

# TODO: increase figure size
# plot_intercepts(x2, a, n, step_function_coefficients, "step_function",
#                 step_function, 0, save=True)
#
# plot_intercepts(x2, a, n, v_function_coefficients, "v_function",
#                 v_function, 1, save=True)

# plot_intercepts_loglog(near_left_a, a, xi, n,
#                        step_function_coefficients,
#                        "step_function",
#                        step_function,
#                        0,
#                        label=r"a-\xi",
#                        name="near_left_a",
#                        save=True)
#
# plot_intercepts_loglog(near_right_a, a, xi, n,
#                        step_function_coefficients,
#                        "step_function",
#                        step_function,
#                        0,
#                        label=r"a+\xi",
#                        name="near_right_a",
#                        save=True)
#
# plot_intercepts_loglog(near_left_edge, a, xi, n,
#                        step_function_coefficients,
#                        "step_function",
#                        step_function,
#                        0,
#                        label=r"-1+\xi",
#                        name="near_left_edge",
#                        save=True)
#
# plot_intercepts_loglog(near_right_edge, a, xi, n,
#                        step_function_coefficients,
#                        "step_function",
#                        step_function,
#                        0,
#                        label=r"1+\xi",
#                        name="near_right_edge",
#                        save=True)
#
# plot_intercepts_loglog(near_left_a, a, xi, n,
#                        v_function_coefficients,
#                        "v_function",
#                        v_function,
#                        1,
#                        label=r"a-\xi",
#                        name="near_left_a",
#                        save=True)
#
# plot_intercepts_loglog(near_right_a, a, xi, n,
#                        v_function_coefficients,
#                        "v_function",
#                        v_function,
#                        1,
#                        label=r"a+\xi",
#                        name="near_right_a",
#                        save=True)
#
# plot_intercepts_loglog(near_left_edge, a, xi, n,
#                        v_function_coefficients,
#                        "v_function",
#                        v_function,
#                        1,
#                        label=r"-1+\xi",
#                        name="near_left_edge",
#                        save=True)
#
# plot_intercepts_loglog(near_right_edge, a, xi, n,
#                        v_function_coefficients,
#                        "v_function",
#                        v_function,
#                        1,
#                        label=r"1+\xi",
#                        name="near_right_edge",
#                        save=True)
