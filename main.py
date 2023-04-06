import matplotlib.pyplot as plt
import numpy as np


def integrate(x: np.ndarray, y, retarray=True):
    dx =  x[1] - x[0]

    sum = 0
    if retarray:
        y_int = []
        for y_i in y:
            sum += dx * y_i
            y_int.append(sum)

        return np.array(y_int)
    else:
        for y_i in y:
            sum += dx * y_i
        return sum


def dft(x: np.ndarray, y: np.ndarray):

    n = len(x)
    df = (x[-1]-x[0])/n

    f = np.linspace(0, 1/(2*df), n)

    F = []
    for f_i in f:
        alpha = integrate(x, y*np.cos(f_i*x), retarray=False)
        beta = integrate(x, y*np.sin(f_i*x), retarray=False)

        F.append(np.sqrt(alpha**2 + beta**2))

    return f, F

def function(x):
    return np.sin(2*x) + 2*np.cos(5*x) + np.cos(20*x + 1) + np.sin(110*x + 2) + 1.5*np.cos(50*x)

def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #   S E T   U P   # # # # # # # # # # # # # # # # # # # # # # # # #
    # Number of samples per unit
    f_s = 500

    # x's increment
    dx = 1/f_s

    # Generate a domain
    x = np.arange(0, 1*np.pi, dx)

    # Generate a function
    y = np.array([function(x) for x in x])

    # Fourier transform
    f, F = dft(x, y)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #   P L O T   # # # # # # # # # # # # # # # # # # # # # # # # # #
    fig, ax = plt.subplots(2, 1)

    # Function
    ax[0].scatter(x, y, s=0.2)
    # ax[0].axis('equal')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    ax[0].grid()

    # Fourier transform
    ax[1].plot(f, F)
    ax[1].set_xlabel('f')
    ax[1].set_ylabel('F(f)')
    ax[1].grid()

    plt.show()


if __name__ == '__main__':
    main()
