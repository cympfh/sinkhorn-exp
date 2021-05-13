import numpy
import streamlit as sl

"""# Random Test"""
n = sl.slider("size", min_value=1, max_value=20)


@sl.cache
def random_problem(size: int):
    """Random Transportation Problem to solve"""
    a = numpy.random.rand(size)
    a /= a.sum()
    b = numpy.random.rand(size)
    b /= b.sum()
    c = numpy.random.rand(size, size)
    c /= c.sum()
    return a, b, c


a, b, c = random_problem(n)
"""a, b"""
sl.write(a)
sl.write(b)

"""C"""
sl.write(c)


def solve(a, b, c, lm: float, loop_times: int):
    """Transportation Problem, 最適輸送問題

    Parameters
    ----------
    a
        source vector
    b
        destination vector
    c
        cost matrix
        c[i][j] is the cost from a[i] to b[j]
    lm
        hyper parameter, lambda
    loop_times
        loop times for convergence
    """
    n = len(a)
    assert lm > 0.0
    k = numpy.exp(c * (-lm))
    u = numpy.ones(n)
    for _ in range(loop_times):
        v = b / (numpy.transpose(k) @ u)
        u = a / (k @ v)
    p = numpy.transpose(u * numpy.transpose(k * v))
    return p


lm = sl.slider("lambda", min_value=0.1, max_value=100.0, step=0.1)
sl.write("lambda=", lm)
loop_times = sl.slider("loop_times", min_value=10, max_value=100)
p = solve(a, b, c, lm, loop_times)
sl.write("P", p)
sl.write("cost", numpy.sum(c * p))


"""# Sorting Problem"""
n = sl.slider("array size", min_value=1, max_value=20)
lm = sl.slider("lambda!", min_value=1, max_value=2000)

@sl.cache
def random_array(size: int):
    """Sorting this x"""
    x = numpy.random.rand(n)
    return x


def my_sort(x, lm: float):
    """Sorting with Transportation Problem Solver"""
    n = len(x)
    y = numpy.array(range(n))
    a = numpy.ones(n) / n
    b = numpy.ones(n) / n
    c = numpy.array([[(x[i] - y[j]) ** 2.0 for j in range(n)] for i in range(n)])
    c /= c.sum()
    p = solve(a, b, c, lm, 100)
    return p


x = random_array(n)
sl.write("sorting x:", x)

p = my_sort(x, lm)
sl.write("loss", numpy.sum(c * p))

s = n * numpy.transpose(p) @ x
sl.write("Sorted(x)", s)

acc = numpy.array(range(1, n + 1)) / n
r = n * n * p @ acc
sl.write("Rank(x)", r)
