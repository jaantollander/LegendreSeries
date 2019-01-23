import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, legendre_series, \
    step_function, v_function, pointwise_convergence

# Improved plot styles.
seaborn.set()

dirname = "figures"


def plot_legendre_polynomials(x, n=5, name="legendre_polynomials", save=False):
    """Plot Legendre polynomials."""
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel("$P_n(x)$")
    p = legendre_polynomials(x)
    for _ in range(n):
        plt.plot(x, next(p))
    if save:
        os.makedirs(dirname, exist_ok=True)
        filepath = os.path.join(dirname, f"{name}.png")
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


def plot_piecewise_functions(x, a, name="piecewise_functions", save=False):
    """Plot Step and V-function."""
    plt.figure()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.plot(x, v_function(x, a), label="$u(x)$")
    plt.plot(x, step_function(x, a), label="$u'(x)$")
    plt.legend()
    if save:
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(os.path.join(dirname, f"{name}.png"), dpi=300)
    else:
        plt.show()


def animate_legendre_series(x, a, n, coeff_fun, name, f, save=False):
    """Create animation of the Legendre series."""
    series = legendre_series(x, coeff_fun(a))

    # Legendre Series
    start = np.min(x)
    stop = np.max(x)
    ymin = np.min(f(x, a)) - 0.3
    ymax = np.max(f(x, a)) + 0.3

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
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

    anim = FuncAnimation(fig, update, frames=n, interval=100)
    if save:
        os.makedirs(dirname, exist_ok=True)
        fpath = os.path.join(dirname, f'{name}.mp4')
        anim.save(fpath, dpi=300, writer='ffmpeg')
        # anim.save(os.path.join(dirname, f'{name}.gif'), dpi=80, writer='imagemagick')
    else:
        plt.show()


def plot_pointwise_convergence(x, a, n, coeff_fun, name, f, beta, save=False):
    """Plot poinwise convergence of Legendre series."""
    degrees, errors, indices, slope, intercept = pointwise_convergence(
        x, a, n, coeff_fun, f, beta)

    plt.figure()
    plt.ylim(1e-8, 1e1)
    plt.title(f"x={x}, a={a}")
    plt.xlabel(r"Degree $n$")
    plt.ylabel(r"Error $\varepsilon(x)$")
    plt.loglog(degrees[1:], errors[1:])
    plt.loglog(degrees[indices], errors[indices])
    plt.loglog(degrees[1:], (10 ** intercept) * degrees[1:] ** slope,
               label=f"Slope: {slope:.2f}\nIntercept: {10 ** intercept:.2f}")
    plt.legend()
    if save:
        fpath = os.path.join(dirname, "pointwise_convergence", name)
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"a{a}_x{x}.png"), dpi=300)
    else:
        plt.show()


def animate_pointwise_convergence():
    """Create an animation of pointwise convergences."""
    pass


def plot_intercepts(xs, a, n, coeff_fun, func_name, f, beta, save=False):
    """Create a plot of the behaviour of the intercepts."""
    intercepts = []
    for x in xs:
        print(x)
        degrees, errors, indices, slope, intercept = pointwise_convergence(
            x, a, n, coeff_fun, f, beta)
        intercepts.append(intercept)

    plt.figure()
    plt.xlabel(r"$x$")
    plt.ylabel(r"Slope $k$")
    plt.plot(xs, intercepts, '.')

    if save:
        fpath = os.path.join(dirname, "intercepts", func_name)
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{a}.png"))
    else:
        plt.show()


def plot_intercepts_loglog(xs, a, xi, n, coeff_fun, func_name, f, beta, label, name, save=False):
    """Create a plot of the behaviour of the intercepts near the singularity
    and edges."""
    intercepts = []
    for x in xs:
        print(x)
        degrees, errors, indices, slope, intercept = pointwise_convergence(
            x, a, n, coeff_fun, f, beta)
        intercepts.append(intercept)

    # Fit a line
    xi_log = np.log10(xi)
    z = np.polyfit(xi_log, intercepts, 1)
    p = np.poly1d(z)

    plt.figure()
    plt.xlabel(r"$\xi$")
    plt.ylabel(f"$k({label})$")
    plt.loglog(xi, 10 ** np.array(intercepts), '.')
    # TODO: improve label, variable names
    plt.loglog(xi, 10 ** p(xi_log), label=f"{z[0]:.5f}\n{z[1]:.5f}")
    plt.legend()

    if save:
        fpath = os.path.join(dirname, "intercepts_loglog", func_name, str(a))
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{name}.png"))
    else:
        plt.show()
