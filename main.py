# Neevin Kirill
# variant number: 12

# f(x) = x^3 - 4,5x^2 - 9,21x - 0,383

# 3 - Метод Ньютона
# 5 - Метод простой итерации
# 6 - Метод Ньютона

from math import sin, exp

from solvers import horde_method, newton_method, simple_iteration_method, system_newton_method, iterative_newton
from graph import show_2d

A, B, C, D = 1, -4.5, -9.21, -0.383
y = lambda x: A * x ** 3 + B * x ** 2 + C * x + D
dy = lambda x: 3*A * x ** 2 + 2*B * x + C


def phi(x):
    base = -(B * x ** 2 + C * x + D) / A
    if base < 0:
        return -(-base) ** (1 / 3)
    return base ** (1 / 3)


def non_linear():
    left, right = map(float, input('Границы левого корня через пробел (-2 -1): ').split())
    c_x0 = float(input('Нулевое приближение центрального корня (0): '))
    r_x0 = float(input('Нулевое приближение правого корня (5): '))
    epsilon = float(input('Погрешность (0.01): '))

    left_x = horde_method(y, left=left, right=right, epsilon=epsilon)
    center_x = newton_method(y, dy, c_x0, epsilon=epsilon)
    right_x = simple_iteration_method(y, phi, r_x0, epsilon=epsilon)

    show_2d(y, [(left_x.root, 0), (center_x.root, 0), (right_x.root, 0)])

    print(f'Левый корень x_0={left_x.root:.3f}')
    print(f'Центральный корень x_1={center_x.root:.3f}')
    print(f'Правый корень x_2={right_x.root:.3f}')

    print(f'Таблица для метода хорд:')
    print(left_x)

    print('Таблица для метода Ньютана:')
    print(center_x)

    print('Таблица для метода простой итерации:')
    print(right_x)


def jacobian_example(xy):
    x, y = xy
    return [
        [1, 2],
        [2*x, 8*y]
    ]

def function_example(xy):
    x, y = xy
    return [
        (-1)*(x+(2*y)-2),
        (-1)*((x**2)+(4*(y**2))-4)
    ]

FUNCTIONS = [
    {
        'disp': 'f1(x1, x2) = x + 2*y - 2\nf2(x1, x2) = x^2 + 4*y^2 - 4)',
        'func': function_example,
        'jacob': jacobian_example,
    },
    {
        'disp': 'f1(x1, x2) = x + 2*y - 2\nf2(x1, x2) = x^2 + 4*y^2 - 4)',
        'func': function_example,
        'jacob': jacobian_example,
    },
]

def non_linear_system():
    for i, group in enumerate(FUNCTIONS, 1):
        print(f'Выберите функцию №{i}')
        print(group['disp'])
    n = int(input())
    funcs = FUNCTIONS[n - 1]['func']
    jacob = FUNCTIONS[n - 1]['jacob']

    x0 = list(map(float, input('Начальные приближения (1, 2): ').split()))
    eps = float(input('Погрешность (0.001):'))

    res = system_newton_method(fun=funcs, jacobian=jacob, x_init=x0, epsilon=eps)
    print(res)

    # if res.solved:
    #     print('Решение: ' + ' '.join(f'{x:.3f}' for x in res.roots))
    #     print('Погрешности: ' + ' '.join(str(x) for x in res.errors))
    #     print(f'Количество итераций {res.iteration}')
    # else:
    #     print('Решений не найдено!')

    # bx = abs(min(res.roots)) + 1
    # show_graph_3d(bx, 0, y, [])


def main():
    # non_linear()
    non_linear_system()


if __name__ == '__main__':
    main()
