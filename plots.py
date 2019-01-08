import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, step_function_coefficients, \
    legendre_series

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
    plt.plot(x, np.abs(x-a), label="$u(x)$")
    plt.plot(x, np.sign(x-a), label="$u'(x)$")
    plt.legend()
    plt.savefig(os.path.join(dirname, "piecewise_functions.png"), dpi=300)
    # plt.show()


def plot_pointwise_convergence(x, a, n):
    coefficients = step_function_coefficients(a)
    series = legendre_series(x, coefficients)
    degs = np.arange(n)
    values = np.array([next(series) for _ in degs])
    error = np.abs(step_function(x, a) - values)
    plt.figure()
    plt.xlabel("$n$")
    plt.ylabel(r"$\varepsilon(x)$")
    plt.loglog(degs, error)
    # plt.legend()
    # plt.savefig(os.path.join(dirname, "pointwise_convergence.png"), dpi=300)
    plt.show()


def animation_legendre_series():
    # Legendre Series
    # TODO: beta parameter
    # TODO: function animation
    start = -1
    stop = 1
    x = np.linspace(start, stop, 1001)
    a = 0.5
    n = 100  # frames

    coefficients = step_function_coefficients(a)
    series = legendre_series(x, coefficients)

    # TODO: title, current degree (n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set(
        xlim=(start, stop),
        ylim=(-1.3, 1.3),
        xlabel="$x$",
        ylabel="$f(x)$",
    )
    axes[1].set(
        xlim=(start, stop),
        ylim=(1e-6, 1),
        xlabel="$x$",
        ylabel=r"$\varepsilon(x)$",
    )
    fig.set_tight_layout(True)
    y = next(series)
    plot_series, = axes[0].plot(x, y)
    plot_error, = axes[1].semilogy(x, np.abs(np.sign(x - a) - y))

    def update(i):
        print(i)
        y = next(series)
        plot_series.set_data(x, y)
        plot_error.set_data(x, np.abs(np.sign(x - a) - y))
        return plot_series

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=n, interval=100)
    anim.save('series.gif', dpi=80, writer='imagemagick')
    # plt.savefig("one_euro_filter.png", dpi=300)
    # plt.show()


plot_pointwise_convergence(0.49, 0.5, 1e5)
