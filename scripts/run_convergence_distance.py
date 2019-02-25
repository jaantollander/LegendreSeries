import numpy as np

from legendre_series import step_function_coefficients, step_function, \
    v_function_coefficients, v_function
from plots import plot_convergence_distance, \
    plot_convergence_distance_loglog


left_edge = -1.0
right_edge = 1.0
a = 0.50  # Singularity TODO: 0.0, ..., 0.5, ..., 0.9
x = np.linspace(left_edge, right_edge, 201)
special_points = [-a, a, 0.0, left_edge, right_edge]
x = np.union1d(x, special_points)
n = 10 ** 5
# TODO: dependent on a, different xi for near a and near edges
xi = np.logspace(-4, -2, 25, base=10)
near_left_a = a - xi
near_right_a = a + xi
near_left_edge = left_edge + xi
near_right_edge = right_edge - xi

# Add the extra points near singularity (a) and edges (-1 and 1).
x2 = np.union1d(
    np.union1d(near_left_a, near_right_a),
    np.union1d(near_left_edge, near_right_edge)
)
x2 = np.union1d(x, x2)


plot_convergence_distance(x2, a, n, step_function_coefficients, "step_function",
                          step_function, 0, save=True)

plot_convergence_distance(x2, a, n, v_function_coefficients, "v_function",
                          v_function, 1, save=True)


plot_convergence_distance_loglog(near_left_a, a, xi, n,
                                 step_function_coefficients,
                                 "step_function",
                                 step_function,
                                 0,
                                 label=r"a-\xi",
                                 name="near_left_a",
                                 save=True)

plot_convergence_distance_loglog(near_right_a, a, xi, n,
                                 step_function_coefficients,
                                 "step_function",
                                 step_function,
                                 0,
                                 label=r"a+\xi",
                                 name="near_right_a",
                                 save=True)

plot_convergence_distance_loglog(near_left_edge, a, xi, n,
                                 step_function_coefficients,
                                 "step_function",
                                 step_function,
                                 0,
                                 label=r"-1+\xi",
                                 name="near_left_edge",
                                 save=True)

plot_convergence_distance_loglog(near_right_edge, a, xi, n,
                                 step_function_coefficients,
                                 "step_function",
                                 step_function,
                                 0,
                                 label=r"1+\xi",
                                 name="near_right_edge",
                                 save=True)

plot_convergence_distance_loglog(near_left_a, a, xi, n,
                                 v_function_coefficients,
                                 "v_function",
                                 v_function,
                                 1,
                                 label=r"a-\xi",
                                 name="near_left_a",
                                 save=True)

plot_convergence_distance_loglog(near_right_a, a, xi, n,
                                 v_function_coefficients,
                                 "v_function",
                                 v_function,
                                 1,
                                 label=r"a+\xi",
                                 name="near_right_a",
                                 save=True)

plot_convergence_distance_loglog(near_left_edge, a, xi, n,
                                 v_function_coefficients,
                                 "v_function",
                                 v_function,
                                 1,
                                 label=r"-1+\xi",
                                 name="near_left_edge",
                                 save=True)

plot_convergence_distance_loglog(near_right_edge, a, xi, n,
                                 v_function_coefficients,
                                 "v_function",
                                 v_function,
                                 1,
                                 label=r"1+\xi",
                                 name="near_right_edge",
                                 save=True)
