import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, step_function_coefficients, \
    legendre_series, v_function_coefficients

dirname = "figures"
os.makedirs(dirname, exist_ok=True)
seaborn.set()


def step_function(x, a, c=1, beta=0):
    return c*np.sign(x-a) + beta


def v_function(x, a, c=1, alpha=0, beta=0):
    return c*np.abs(x-a) + alpha + beta*x


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


def plot_pointwise_convergence(x, a, n, coefficients, name, f):
    series = legendre_series(x, coefficients(a))

    assert isinstance(x, float)
    degs = np.arange(n)
    values = np.array([next(series) for _ in degs])
    error = np.abs(f(x, a) - values)

    plt.figure()
    plt.title(f"x={x}, a={a}")
    plt.xlabel(r"Degree $n$")
    plt.ylabel(r"Error $\varepsilon(x)$")
    plt.loglog(degs, error)
    plt.savefig(os.path.join(dirname, f"pointwise_convergence_{name}_{x}_{a}.png"), dpi=300)
    plt.show()


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


# TODO: beta parameter
x = np.linspace(-1, 1, 1001)
a = 0.5
n = 200  # frames
animate_legendre_series(x, a, n,
                        step_function_coefficients(a),
                        f"step_function_series_{n}",
                        step_function)

animate_legendre_series(x, a, n,
                        v_function_coefficients(a),
                        f"v_function_series_{n}",
                        v_function)
