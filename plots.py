import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, step_function_coefficients, \
    legendre_series, step_function, v_function


dirname = "figures"
os.makedirs(dirname, exist_ok=True)
seaborn.set()


def plot_legendre_polynomials(x, a):
    # Legendre polynomials plot
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel("$P(x)$")
    p = legendre_polynomials(x)
    for _ in range(5):
        y = next(p)
        plt.plot(x, y)
    plt.savefig(os.path.join(dirname, "legendre_polynomials.png"), dpi=300)
    # plt.show()


def plot_piecewise_functions(x, a):
    # Step and V-function plots
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.plot(x, v_function(x, a), label="$u(x)$")
    plt.plot(x, step_function(x, a), label="$u'(x)$")
    plt.legend()
    plt.savefig(os.path.join(dirname, f"piecewise_functions.png"), dpi=300)
    # plt.show()


def animate_legendre_series(x, a, n, coefficients, name, f):
    series = legendre_series(x, coefficients(a))

    # Legendre Series
    start = np.min(x)
    stop = np.max(x)
    ymin = np.min(f(x, a)) - 0.3
    ymax = np.max(f(x, a)) + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set(
        xlim=(start, stop),
        ylim=(ymin, ymax),
        xlabel="$x$",
        ylabel="$f(x)$",
    )
    axes[1].set(
        xlim=(start, stop),
        ylim=(1e-6, 1.1),
        xlabel="$x$",
        ylabel=r"$\varepsilon(x)$",
    )
    axes[0].set_title(f"n={0}")
    axes[0].plot(x, f(x, a))
    fig.set_tight_layout(True)
    y = next(series)
    plot_series, = axes[0].plot(x, y)
    error = np.abs(f(x, a) - y)
    plot_error, = axes[1].semilogy(x, error)

    def update(i):
        print(i)
        y = next(series)
        axes[0].set_title(f"n={i}")
        plot_series.set_data(x, y)
        error = np.abs(f(x, a) - y)
        plot_error.set_data(x, error)
        return plot_series, plot_error

    # TODO: mp4?
    anim = FuncAnimation(fig, update, frames=n, interval=100)
    anim.save(os.path.join(dirname, f'{name}.gif'), dpi=80, writer='imagemagick')
    # plt.show()


def plot_pointwise_convergence(x, a, n, coefficients, name, f):
    series = legendre_series(x, coefficients(a))

    assert isinstance(x, float)
    degs = np.arange(n)
    values = np.array([next(series) for _ in degs])
    error = np.abs(f(x, a) - values)

    d = np.log10(degs)
    e = np.log10(error)

    # Convergence slopes
    # TODO: verify for correctness
    i = 1  # Start from one
    indices = [i]
    while i < len(d)-1:
        i = np.argmax((e[i + 1:] - e[i]) / (d[i + 1:] - d[i])) + 1 + i
        indices.append(i)

    # TODO: calculate and plot convergence line and label
    plt.figure()
    plt.title(f"x={x}, a={a}")
    plt.xlabel(r"Degree $n$")
    plt.ylabel(r"Error $\varepsilon(x)$")
    plt.loglog(degs[1:], error[1:])
    plt.loglog(degs[indices], error[indices])
    fpath = os.path.join(dirname, "pointwise_convergence", name)
    os.makedirs(fpath, exist_ok=True)
    # plt.savefig(os.path.join(fpath, f"x{x}_a{a}.png"), dpi=300)
    plt.show()


# TODO: beta parameter
# TODO: convergence values
x = np.linspace(-1, 1, 1001)
a = 0.5
n = 200  # frames
# animate_legendre_series(x, a, n,
#                         step_function_coefficients,
#                         f"step_function_series_{n}",
#                         step_function)
#
# animate_legendre_series(x, a, n,
#                         v_function_coefficients,
#                         f"v_function_series_{n}",
#                         v_function)

# TODO: pointwise convergence near edges and singularity
# TODO: function animation of pointwise convergence
plot_pointwise_convergence(0.8999, 0.9, 1e5,
                           step_function_coefficients,
                           "step_function",
                           step_function)
