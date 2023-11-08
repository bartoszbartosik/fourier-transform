import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi


def dft(x: np.ndarray, y: np.ndarray):
    """
    Perform a Discrete Fourier Transform on a y(x) function.
    """
    # Number of samples
    N = len(x)

    # Frequency resolution (sampling frequency reciprocal)
    df = 1 / x[-1] * 2*pi

    # Frequency domain
    f = np.linspace(0, N * df, N)

    # Dictionary for storing computations results
    results = {
        'f': f,
        're': np.array([]),
        'im': np.array([])
    }

    for k in range(N):
        re = 0
        im = 0
        for n in range(N):
            re += y[n]*cos(2*pi/N*k*n)
            im += y[n]*sin(2*pi/N*k*n)
        results['re'] = np.append(results['re'], re)
        results['im'] = np.append(results['im'], im)

    return results


def ift(dft: dict):
    """
    Perform an Inverse Discrete Fourier Transform.
    """
    # Extract frequency, real, and imaginary part
    f, re, im = dft.values()

    # Number of samples
    N = len(f)

    # Original function's domain resolution
    dx = 2*pi / f[-1]

    # Original function domain
    x = np.linspace(0, N * dx, N)

    # Dictionary for storing computations results
    results = {
        'x': x,
        're': np.array([]),
        'im': np.array([])
    }

    for n in range(N):
        re_i = 0
        im_i = 0
        for k in range(N):
            re_i += re[k]*cos(2*pi/N*k*n) + im[k]*sin(2*pi/N*k*n)
            im_i += re[k]*sin(2*pi/N*k*n) - im[k]*cos(2*pi/N*k*n)
        results['re'] = np.append(results['re'], re_i/N)
        results['im'] = np.append(results['im'], re_i/N)

    return results


# Define a function to be analysed
def function(x):
    # return sin(2*x) + 2*cos(5*x) + cos(20*x + 1.71) + sin(110*x -1.71) + 1.5*cos(50*x)
    # return 2*sin(2*x - pi/4) + 2*cos(5*x) + 3*cos(20*x) + 4*sin(110*x) + 0.2*cos(50*x)
    return 2*sin(50*x - pi/4)
    # return sin(100*x + pi/4)


# Main function
def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #   S E T   U P   # # # # # # # # # # # # # # # # # # # # # # # # #
    # Sampling frequency (number of samples per unit)
    f_s = 500

    # Function interval
    x_end = pi/4

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Function domain increment
    dx = 1/f_s

    # Generate a domain
    x = np.arange(0, x_end, dx)

    # Generate a function
    y = np.array([function(x) for x in x])

    # Compute Fourier transform
    dtft_dict = dft(x, y)

    # Compute inverse Fourier transform
    ift_dict = ift(dtft_dict)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   D A T A   P O S T P R O C E S S I N G   # # # # # # # # # # # # # # # # # #
    # Extract frequency, real and imaginary part, subsequently
    f, re, im = dtft_dict.values()

    # Compute each frequency magnitude
    amp = 1/len(f)*np.sqrt(re**2 + im**2)

    # Compute each frequency phase shift
    phi = np.arctan2(re, im)*180/pi

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Extract real and imaginary part of original signal
    x_i, re_i, im_i = ift_dict.values()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #   P L O T   # # # # # # # # # # # # # # # # # # # # # # # # # #
    plot_data = {
        'xs': [x, f, f, f, f, x_i],
        'ys': [y, amp, phi, re, im, re_i],
        'plot_types': [plt.scatter, plt.stem, plt.stem, plt.stem, plt.stem, plt.scatter],
        'subplots': [411, 423, 424, 425, 426, 414],
        'titles': ['Analysed function', 'Amplitude', 'Phase shift', 'Real part', 'Imaginary part', 'Reconstructed signal'],
        'x_labels': ['x', 'f', 'f', 'f', 'f', 'x'],
        'y_labels': ['y(x)', 'A(f)', '\u03C6(f)', 'Re(f)', 'Im(f)', 'y\'(x)']
    }

    for i in range(len(plot_data['xs'])):
        plt.subplot(plot_data['subplots'][i])
        if plot_data['plot_types'][i].__name__ == 'scatter':
            plot_data['plot_types'][i](plot_data['xs'][i], plot_data['ys'][i], s=0.2, c='0.3')
        else:
            markerline, stemline, baseline = plot_data['plot_types'][i](plot_data['xs'][i], plot_data['ys'][i], linefmt='0.3', markerfmt='o')
            markerline.set_markerfacecolor('0.3')
            markerline.set_markeredgecolor('0.3')
            baseline.set_color('none')
            stem_linewidth = 1
            stem_markersize = 2
            plt.setp(stemline, linewidth=stem_linewidth)
            plt.setp(markerline, markersize=stem_markersize)
        plt.title(plot_data['titles'][i])
        plt.xlabel(plot_data['x_labels'][i])
        plt.ylabel(plot_data['y_labels'][i])
        plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
