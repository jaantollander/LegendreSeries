import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.animation import FuncAnimation

from legendre_series import legendre_polynomials, legendre_series, \
    step_function, v_function, convergence_rate, convergence_line_log

# Improved plot styles.
seaborn.set()

# TODO: plot_legendre_series


def plot_legendre_polynomials(x, n=5, name="legendre_polynomials", save=False,
                              dirname="figures"):
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


def plot_piecewise_functions(x, a, name="piecewise_functions", save=False,
                             dirname="figures"):
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


def plot_legendre_series(x, a, n, coeff_func, name, f, ylim_min,
                         save=False, dirname="figures"):
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
        ylabel="$f_k(x)$",
    )
    axes[1].set(
        xlim=(start, stop),
        ylim=(ylim_min, 1.1),
        xlabel="$x$",
        ylabel=r"$|\varepsilon_k(x)|$",
    )
    axes[0].set_title(f"k={n}")
    axes[1].set_title(f"k={n}")
    axes[0].plot(x, f(x, a))
    fig.set_tight_layout(True)

    for _ in range(n):
        next(series)

    y = next(series)
    plot_series, = axes[0].plot(x, y)
    error = np.abs(f(x, a) - y)
    plot_error, = axes[1].semilogy(x, error)

    if save:
        os.makedirs(dirname, exist_ok=True)
        plt.savefig(os.path.join(dirname, f"legendre_series.png"), dpi=300)
    else:
        plt.show()
    plt.close(fig)


def animate_legendre_series(x, a, n, coeff_func, name, f, ylim_min,
                            save=False, dirname="figures"):
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
        ylabel="$f_k(x)$",
    )
    axes[1].set(
        xlim=(start, stop),
        ylim=(ylim_min, 1.1),
        xlabel="$x$",
        ylabel=r"$|\varepsilon_k(x)|$",
    )
    axes[0].set_title(f"k={0}")
    axes[1].set_title(f"k={0}")
    axes[0].plot(x, f(x, a))
    fig.set_tight_layout(True)
    y = next(series)
    plot_series, = axes[0].plot(x, y)
    error = np.abs(f(x, a) - y)
    plot_error, = axes[1].semilogy(x, error)

    def update(i):
        print(i)
        y = next(series)
        axes[0].set_title(f"k={i}")
        axes[1].set_title(f"k={i}")
        plot_series.set_data(x, y)
        error = np.abs(f(x, a) - y)
        plot_error.set_data(x, error)
        return plot_series, plot_error

    anim = FuncAnimation(fig, update, frames=n, interval=100)
    if save:
        # TODO: {function}/{a}
        os.makedirs(dirname, exist_ok=True)
        fpath = os.path.join(dirname, f'{name}.mp4')
        anim.save(fpath, dpi=300, writer='ffmpeg')
        # anim.save(os.path.join(dirname, f'{name}.gif'), dpi=80, writer='imagemagick')
    else:
        plt.show()
    plt.close(fig)


def plot_pointwise_convergence(x, a, n, coeff_func, name, f, b, ylim_min,
                               save=False, dirname="figures"):
    """Plot poinwise convergence of Legendre series."""
    series = legendre_series(x, coeff_func(a))
    degrees = np.arange(n)
    values = np.array([next(series) for _ in degrees])
    errors = np.abs(f(x, a) - values)

    a_min = -convergence_rate(x, a, b)
    alpha, beta = convergence_line_log(degrees, errors, a_min)

    fig, ax = plt.subplots()
    ax.set(
        ylim=(ylim_min, 1e1),
        title=f"x={x}, a={a}",
        xlabel=r"$k$",
        ylabel=r"$|\varepsilon_k(x)|$"
    )
    ax.loglog(degrees[1:], errors[1:])
    # ax.loglog(degrees[indices], errors[indices])
    ax.loglog(degrees[1:], beta * degrees[1:] ** alpha,
              label=rf"$\alpha={-alpha:.3f}$"+'\n'+rf"$\beta={beta:.3f}$")
    ax.legend()
    if save:
        fpath = os.path.join(dirname, "pointwise_convergence", name, str(a))
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{x:.7f}.png"), dpi=300)
    else:
        plt.show()
    plt.close(fig)


def animate_pointwise_convergence(dirname="figures"):
    """Create an animation of pointwise convergences."""
    pass


def plot_convergence_distance(xs, a, n, coeff_func, func_name, f, b, save=False,
                              dirname="figures"):
    """Create a plot of the behaviour of the intercepts."""
    betas = []
    for x in xs:
        print(x)
        series = legendre_series(x, coeff_func(a))
        degrees = np.arange(n)
        values = np.array([next(series) for _ in degrees])
        errors = np.abs(f(x, a) - values)

        a_min = -convergence_rate(x, a, b)
        alpha, beta = convergence_line_log(degrees, errors, a_min)
        betas.append(beta)

    fig = plt.figure(figsize=(16, 8))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\beta(x)$")
    plt.plot(xs, np.log10(betas), '.')

    if save:
        fpath = os.path.join(dirname, "convergence_distances", func_name)
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{a}.png"))
    else:
        plt.show()
    plt.close(fig)


def plot_convergence_distance_loglog(xs, a, xi, n, coeff_func, func_name, f, b,
                                     label, name, save=False, dirname="figures"):
    """Create a plot of the behaviour of the intercepts near the singularity
    and edges."""
    betas = []
    for x in xs:
        print(x)
        series = legendre_series(x, coeff_func(a))
        degrees = np.arange(n)
        values = np.array([next(series) for _ in degrees])
        errors = np.abs(f(x, a) - values)

        a_min = -convergence_rate(x, a, b)
        alpha, beta = convergence_line_log(degrees, errors, a_min)
        betas.append(beta)

    # Fit a line
    xi_log = np.log10(xi)
    z = np.polyfit(xi_log, np.log10(betas), 1)
    p = np.poly1d(z)

    fig = plt.figure()
    plt.xlabel(r"$\xi$")
    plt.ylabel(rf"$\beta({label})$")
    plt.loglog(xi, np.array(betas), '.', label=r"$\beta$")
    # TODO: improve label, variable names
    plt.loglog(xi, 10 ** p(xi_log),
               label="\n".join((rf"$\rho={-z[0]:.5f}$", rf"$D={10**z[1]:.5f}$")))
    plt.legend()

    if save:
        fpath = os.path.join(dirname, "convergence_distances_loglog", func_name, str(a))
        os.makedirs(fpath, exist_ok=True)
        plt.savefig(os.path.join(fpath, f"{name}.png"))
    else:
        plt.show()
    plt.close(fig)
