import numpy as np


def step_function(x, a, c=1, beta=0):
    """Step function."""
    return c*np.sign(x-a) + beta


def v_function(x, a, c=1, alpha=0, beta=0):
    """V-function."""
    return c*np.abs(x-a) + alpha + beta*x


def legendre_polynomials(x):
    """Generator of Legendre polynomials at x."""
    assert isinstance(x, (float, np.ndarray))
    p0 = 1.0 if isinstance(x, float) else np.ones_like(x)
    p1 = x
    yield p0
    yield p1
    n = 2
    while True:
        p2 = ((2*n-1)*x*p1 - (n-1)*p0)/n
        yield p2
        p0 = p1
        p1 = p2
        n += 1


def step_function_coefficients(a):
    """Generator of step function coefficients for Legendre series at a."""
    # TODO: add c and beta parameters
    assert isinstance(a, float)
    p_a1 = legendre_polynomials(a)
    p_a2 = legendre_polynomials(a)
    next(p_a2)
    next(p_a2)
    yield -a
    while True:
        yield next(p_a1) - next(p_a2)


def v_function_coefficients(a):
    """Generator of V-function coefficients for Legendre series at a."""
    # TODO: add c, beta and alpha parameters
    assert isinstance(a, float)
    a_n1 = step_function_coefficients(a)
    a_n2 = step_function_coefficients(a)
    next(a_n1)
    next(a_n1)
    n = 0
    yield (a**2+1)/2
    while True:
        n += 1
        yield -next(a_n1)/(2*n+3) + next(a_n2)/(2*n-1)


def legendre_series(x, coeff_gen):
    """Generator of Legendre series at x for given coefficient generator."""
    assert isinstance(x, (float, np.ndarray))
    p_n = legendre_polynomials(x)
    s = 0
    while True:
        s += next(p_n) * next(coeff_gen)
        yield s


def convergence_rate(x, a, b):
    """Conjecture about the value of the convergence rate."""
    if x == a:
        return max(b, 1)
    elif x == -1 or x == 1:
        return b + 1 / 2
    else:
        return b + 1


def convergence_line(x, y, a_min=-np.inf):
    """Convergence line."""
    assert len(x) == len(y) > 1
    i = 1
    j = i
    while j < len(x)-1:
        k = np.argmax((y[j + 1:] - y[j]) / (x[j + 1:] - x[j])) + 1 + j
        a = (y[k] - y[j]) / (x[k] - x[j])
        if a < a_min:
            break
        i = j
        j = k
    a = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - a*x[i]
    return a, b


def convergence_line_log(x, y, a_min):
    """Convergence line in logarithmic scale."""
    a, b = convergence_line(np.log10(x), np.log10(y), a_min)
    return a, 10**b
