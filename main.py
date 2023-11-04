import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi


def integrate(x: np.ndarray, y, return_array=False):
    """
    Integrate a y(x) function.
    """
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

def dtft(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform a Discrete Fourier Transform on a y(x) function.
    """
    # Number of samples
    n = len(x)

    # Frequency increment (sampling frequency reciprocal)
    df = (x[1]-x[0])

    # Symmetry axis
    f_sym = 1/(2*df)*x[-1]

    # Frequency domain
    f = np.linspace(0, f_sym, n)

    # Compute Fourier Transform
    results = {
        'f': f,
        're': np.array([]),
        'im': np.array([])
    }
    for f_i in f:
        re = integrate(x, y*cos(f_i*x))
        im = integrate(x, y*sin(f_i*x))

        results['re'] = np.append(results['re'], re)
        results['im'] = np.append(results['im'], im)

    return results

# Define a function to be analysed
def function(x):
    return sin(2*x) + 2*cos(5*x) + cos(20*x + 1.71) + sin(110*x -1.71) + 1.5*cos(50*x)

# Main function
def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #   S E T   U P   # # # # # # # # # # # # # # # # # # # # # # # # #
    # Sampling frequency (number of samples per unit)
    f_s = 200

    # Function interval
    x_end = 2

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Function domain increment
    dx = 1/f_s

    # Generate a domain
    x = np.arange(0, x_end, dx)

    # Generate a function
    y = np.array([function(x) for x in x])

    # Compute Fourier transform
    dtft_dict = dtft(x, y)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   D A T A   P O S T P R O C E S S I N G   # # # # # # # # # # # # # # # # # #
    # Extract frequency, real and imaginary part, subsequently
    f, re, im = dtft_dict.values()

    # Compute each frequency magnitude
    mags = np.sqrt(re**2 + im**2)

    # Compute each frequency phase shift
    phi = np.arctan(im/re)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #   P L O T   # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Analysed function
    plt.subplot(311)
    plt.scatter(x, y, s=0.2)
    plt.title('Analysed function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()

    # Magnitude
    plt.subplot(323)
    plt.plot(f, mags)
    plt.title('Magnitude')
    plt.xlabel('f')
    plt.ylabel('M(f)')
    plt.grid()

    # Phase shift
    plt.subplot(324)
    plt.plot(f, phi)
    plt.title('Phase shift')
    plt.xlabel('f')
    plt.ylabel('phi(f)')
    plt.grid()

    # Real part
    plt.subplot(325)
    plt.plot(f, re)
    plt.title('Real part')
    plt.xlabel('f')
    plt.ylabel('re(f)')
    plt.grid()

    # Imaginary part
    plt.subplot(326)
    plt.plot(f, im)
    plt.title('Imaginary part')
    plt.xlabel('f')
    plt.ylabel('im(f)')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
