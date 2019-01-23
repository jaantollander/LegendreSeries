import numpy as np


# TODO: c, alpha and beta parameters


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


def conjecture(x, a, beta):
    """Conjecture about the values of slopes of the pointwise convergence."""
    if x == a:
        return -1 if beta == 0 else -beta
    elif x == -1 or x == 1:
        return -(beta + 1/2)
    else:
        return -(beta + 1)


def pointwise_convergence(x, a, n, coeff_func, f, beta):
    """Computes the pointwise convergence."""
    # FIXME: x=0, a=0
    assert isinstance(x, float)
    assert isinstance(a, float)
    assert isinstance(n, int) and n > 0

    series = legendre_series(x, coeff_func(a))
    degrees = np.arange(n)
    values = np.array([next(series) for _ in degrees])
    errors = np.abs(f(x, a) - values)

    # Logarithmic values
    d = np.log10(degrees)
    e = np.log10(errors)

    # Convergence slopes
    i = 1  # Start from one
    indices = [i]  # Candidates
    lines = []
    while i < len(d)-1:
        j = np.argmax((e[i + 1:] - e[i]) / (d[i + 1:] - d[i])) + 1 + i
        slope = (e[j] - e[i]) / (d[j] - d[i])
        intercept = e[i] - d[i] * slope
        lines.append((slope, intercept))
        indices.append(j)
        i = j

    # Use the conjecture to filter out bad candidates
    l = list(filter(lambda elem: elem[0] > conjecture(x, a, beta), lines))
    slope, intercept = l[-1]

    return degrees, errors, indices, slope, intercept
