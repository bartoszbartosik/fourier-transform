from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

def integrate(function: Callable, domain: np.ndarray, step: float):
    x = domain
    y = np.array([function(x) for x in x])

    sum = 0
    y_int = []
    for y in y:
        sum += step*y
        y_int.append(sum)

    return np.array(y_int)

def ft(function: Callable):
    pass

def function(x):
    return -x+2

def main():

    x, step = np.linspace(-10, 10, 1000, retstep=True)
    y = np.array([function(x) for x in x])
    y_int = integrate(function, x, step)

    # print('integral: {}'.format(integrate(function, x, step)))

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x, y)
    ax[0].axis('equal')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    ax[0].grid()

    ax[1].plot(x, y_int)
    ax[1].axis('equal')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('F(x)')
    ax[1].grid()

    # plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
