from legendre_series import step_function_coefficients, step_function, \
    v_function_coefficients, v_function
from plots import plot_pointwise_convergence


a = 0.5
n = int(1e5)
xs = sorted([-1.0, -0.9, -0.999, -0.9999, -0.99999,
             -0.5, 0.0, 0.49, 0.499, 0.4999,
             0.5, 0.5001, 0.501, 0.51,
             0.999, 0.9999, 0.99999, 1.0])


for x in xs:
    print(x)
    plot_pointwise_convergence(x, a, n, step_function_coefficients,
                               "step_function", step_function, 0,
                               ylim_min=1e-6, save=True)

for x in xs:
    print(x)
    plot_pointwise_convergence(x, a, n, v_function_coefficients,
                               "v_function", v_function, 1,
                               ylim_min=1e-10, save=True)
