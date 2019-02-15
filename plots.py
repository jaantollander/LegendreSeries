import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, legendre_series, \
    step_function, v_function, conjecture, convergence_line_log

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


def animate_legendre_series(x, a, n, coeff_func, name, f, save=False):
    """Create animation of the Legendre series."""
    series = legendre_series(x, coeff_func(a))

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
    plt.close(fig)


def plot_pointwise_convergence(x, a, n, coeff_func, name, f, b, save=False):
    """Plot poinwise convergence of Legendre series."""
    series = legendre_series(x, coeff_func(a))
    degrees = np.arange(n)
    values = np.array([next(series) for _ in degrees])
    errors = np.abs(f(x, a) - values)

    a_min = conjecture(x, a, b)
    alpha, beta = convergence_line_log(degrees, errors, a_min)

    fig, ax = plt.subplots()
    ax.set(
        ylim=(1e-8, 1e1),
        title=f"x={x}, a={a}",
        xlabel=r"Degree $n$",
        ylabel=r"Error $\varepsilon(x)$"
    )
    ax.loglog(degrees[1:], errors[1:])
    # ax.loglog(degrees[indices], errors[indices])
    ax.loglog(degrees[1:], beta * degrees[1:] ** alpha,
              label=f"Slope: {alpha:.2f}\nIntercept: {beta:.2f}")
    ax.legend()
    if save:
        fpath = os.path.join(dirname, "pointwise_convergence", name, str(a))
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{x:.7f}.png"), dpi=300)
    else:
        plt.show()
    plt.close(fig)


def animate_pointwise_convergence():
    """Create an animation of pointwise convergences."""
    pass


def plot_intercepts(xs, a, n, coeff_func, func_name, f, b, save=False):
    """Create a plot of the behaviour of the intercepts."""
    intercepts = []
    for x in xs:
        print(x)
        series = legendre_series(x, coeff_func(a))
        degrees = np.arange(n)
        values = np.array([next(series) for _ in degrees])
        errors = np.abs(f(x, a) - values)

        a_min = conjecture(x, a, b)
        alpha, beta = convergence_line_log(degrees, errors, a_min)
        intercepts.append(beta)

    fig = plt.figure(figsize=(16, 8))
    plt.xlabel(r"$x$")
    plt.ylabel(r"Slope $k$")
    plt.plot(xs, np.log10(intercepts), '.')

    if save:
        fpath = os.path.join(dirname, "intercepts", func_name)
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{a}.png"))
    else:
        plt.show()
    plt.close(fig)


def plot_intercepts_loglog(xs, a, xi, n, coeff_func, func_name, f, b, label, name, save=False):
    """Create a plot of the behaviour of the intercepts near the singularity
    and edges."""
    intercepts = []
    for x in xs:
        print(x)
        series = legendre_series(x, coeff_func(a))
        degrees = np.arange(n)
        values = np.array([next(series) for _ in degrees])
        errors = np.abs(f(x, a) - values)

        a_min = conjecture(x, a, b)
        alpha, beta = convergence_line_log(degrees, errors, a_min)
        intercepts.append(beta)

    # Fit a line
    xi_log = np.log10(xi)
    z = np.polyfit(xi_log, np.log10(intercepts), 1)
    p = np.poly1d(z)

    fig = plt.figure()
    plt.xlabel(r"$\xi$")
    plt.ylabel(f"$k({label})$")
    plt.loglog(xi, np.array(intercepts), '.')
    # TODO: improve label, variable names
    plt.loglog(xi, 10 ** p(xi_log), label=f"{z[0]:.5f}\n{z[1]:.5f}")
    plt.legend()

    if save:
        fpath = os.path.join(dirname, "intercepts_loglog", func_name, str(a))
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{name}.png"))
    else:
        plt.show()
    plt.close(fig)
