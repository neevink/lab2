from typing import Callable, Optional, List

import numpy as np
from attr import dataclass, field
from prettytable import PrettyTable


MAX_ITER_COUNT = 100


@dataclass
class Result:
    root: Optional[float] = None
    error: Optional[float] = None
    header: list = field(factory=list)
    data: list = field(factory=list)

    def __str__(self):
        tt = PrettyTable(self.header)
        data = [(n, *map(lambda x: f'{x:.3f}', floats)) for n, *floats in self.data]
        tt.add_rows(data)
        return str(tt)


@dataclass
class SystemResult:
    iteration: int
    solved: bool
    roots: List[float] = field(factory=list)
    errors: List[float] = field(factory=list)


def horde_method(f: Callable, left: float, right: float, fix=-1, epsilon=10e-3):
    res = Result(header='№ a b x f(a) f(b) f(x) |a-b|'.split())

    x0 = left if fix == -1 else right

    for i in range(1, MAX_ITER_COUNT + 1):
        x1 = (left * f(right) - right * f(left)) / (f(right) - f(left))

        res.data.append([
            i, left, right, x1, f(left), f(right), f(x1), abs(left - right)
        ])

        if abs(x1 - x0) <= epsilon or abs(f(x1)) <= epsilon:
            res.root = x1
            res.error = abs(x1 - x0)
            break
        if f(x1) * f(left) < 0:
            right = x1
        else:
            left = x1
        x0 = x1
    return res


#def newton_method(y: Callable, df: Callable, x0: float, epsilon=10e-3):
#    res = Result(header="№ x_k f(x_k) f'(x_k) x_{k+1} |x_k-x_{k+1}|".split())
#
#    for i in range(1, MAX_ITER_COUNT + 1):
#        x1 = x0 - y(x0) / df(x0)
#        res.data.append([i, x0, y(x0), df(x0), x1, abs(x1 - x0)])
#
#        if abs(x1 - x0) <= epsilon or abs(y(x1) / df(x1)) <= epsilon or abs(y(x1)) <= epsilon:
#            res.root = x1
#            res.error = abs(x1 - x0)
#            break
#
#        x0 = x1
#
#    return res


def simple_iteration_method(f: Callable, phi: Callable, x0=1, epsilon=10e-3):
    res = Result(header="№ x_k f(x_k) x_{k+1} phi(x_k) |x_k-x_{k+1}|".split())

    for i in range(1, MAX_ITER_COUNT + 1):
        x1 = phi(x0)

        res.data.append([
            i, x0, f(x0), x1, phi(x0), abs(x1 - x0)
        ])

        if abs(x1 - x0) <= epsilon:
            res.root = x1
            res.error = abs(x1 - x0)
            break
        x0 = x1
    return res

# Example from the video:
# from youtube https://www.youtube.com/watch?v=zPDp_ewoyhM

# TODO FROM HERE: https://stackoverflow.com/questions/52020775/solving-a-non-linear-system-of-equations-in-python-using-newtons-method


def x_delta_by_gauss(J, b):
    return np.linalg.solve(J, b)

def x_plus_1(x_delta, x_previous):
    x_next = x_previous + x_delta
    return x_next

def newton_method(f, jack, x_init):
    jacobian = jack(*x_init)
    vector_b_f_output = f(*x_init)
    x_delta = x_delta_by_gauss(jacobian, vector_b_f_output)
    x_plus_1 = x_delta + x_init
    return x_plus_1

def iterative_newton(f, jack, x_init, epsilon):
    counter = 0
    x_old = x_init
    x_new = newton_method(f, jack, x_old)
    diff = np.linalg.norm(x_old-x_new)

    while diff > epsilon:
        counter += 1
        x_new = newton_method(f, jack, x_old)
        diff = np.linalg.norm(x_old-x_new)
        x_old = x_new
    convergent_val = x_new
    return convergent_val


def system_newton_method(fun, jacobian, x_init, epsilon=0.001):
    print(list(map(float, (iterative_newton(fun, jacobian, x_init, epsilon)))))
