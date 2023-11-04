import matplotlib.pyplot as plt
import numpy as np

# Integrate an y(x) function
def integrate(x: np.ndarray, y, return_array=False):
    dx =  x[1] - x[0]

    da = 0
    if return_array:
        y_int = []
        for y_i in y:
            da += dx * y_i
            y_int.append(da)

        return np.array(y_int)
    else:
        for y_i in y:
            da += dx * y_i
        return da

# Perform a Discrete Fourier Transform on a y(x) function
# TODO: add phase shift
def dtft(x: np.ndarray, y: np.ndarray):
    # Number of samples
    n = len(x)

    # Frequency increment (sampling frequency reciprocal)
    df = (x[-1]-x[0])/n

    # Symmetry axis
    f_sym = 1/(2*df)*x[-1]

    # Frequency domain
    f = np.linspace(0, f_sym, n)

    # Compute Fourier Transform
    results = {
        'f': f,
        're': [],
        'im': []
    }
    for f_i in f:
        re = integrate(x, y*np.cos(f_i*x))
        im = integrate(x, y*np.sin(f_i*x))

        results['re'].append(re)
        results['im'].append(im)

    return results

# Define a function for which Fourier Transform will be performed
def function(x):
    return np.sin(2*x) + 2*np.cos(5*x) + np.cos(20*x + 1) + np.sin(110*x + 2) + 1.5*np.cos(50*x)

# Main function
def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #   S E T   U P   # # # # # # # # # # # # # # # # # # # # # # # # #
    # Sampling frequency
    f_s = 100

    # x's increment
    dx = 1/f_s

    # Generate a domain
    x = np.arange(0, 2*np.pi, dx)

    # Generate a function
    y = np.array([function(x) for x in x])

    # Fourier transform
    dtft_dict = dtft(x, y)

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
    ax[1].plot(f, ft)
    ax[1].set_xlabel('f')
    ax[1].set_ylabel('F(f)')
    ax[1].grid()

    plt.show()


if __name__ == '__main__':
    main()
