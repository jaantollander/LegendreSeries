import numpy as np


def step_function(x, a, c=1, beta=0):
    return c*np.sign(x-a) + beta


def v_function(x, a, c=1, alpha=0, beta=0):
    return c*np.abs(x-a) + alpha + beta*x


def legendre_polynomials(x):
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
    assert isinstance(a, float)
    p_a1 = legendre_polynomials(a)
    p_a2 = legendre_polynomials(a)
    next(p_a2)
    next(p_a2)
    yield -a
    while True:
        yield next(p_a1) - next(p_a2)


def v_function_coefficients(a):
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


def legendre_series(x, coefficients):
    assert isinstance(x, (float, np.ndarray))
    p_n = legendre_polynomials(x)
    s = 0
    while True:
        s += next(p_n) * next(coefficients)
        yield s


def pointwise_convergence(x, a, n, f):
    pass
