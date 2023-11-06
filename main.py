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


def idft(dtft: dict, x):
    f, re, im = dtft.values()

    # Compute each frequency magnitude
    amp = np.sqrt(re ** 2 + im ** 2)

    # Compute each frequency phase shift
    phi = np.arctan2(re, im)

    s = amp*sin(np.outer(f, x) - phi)

    np.sum(s, axis=0)

    return s

# Define a function to be analysed
def function(x):
    # return sin(2*x) + 2*cos(5*x) + cos(20*x + 1.71) + sin(110*x -1.71) + 1.5*cos(50*x)
    # return 2*sin(2*x - pi/4) + 2*cos(5*x) + 3*cos(20*x) + 4*sin(110*x) + 0.2*cos(50*x)
    # return 2*sin(50*x - pi/4)
    return sin(100*x - pi/4)

# Main function
def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #   S E T   U P   # # # # # # # # # # # # # # # # # # # # # # # # #
    # Sampling frequency (number of samples per unit)
    f_s = 300

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
    dtft_dict = dtft(x, y)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   D A T A   P O S T P R O C E S S I N G   # # # # # # # # # # # # # # # # # #
    # Extract frequency, real and imaginary part, subsequently
    f, re, im = dtft_dict.values()

    # Compute each frequency magnitude
    amp = np.sqrt(re**2 + im**2)

    # Compute each frequency phase shift
    phi = np.arctan2(re, im)*180/pi



    plt.plot(x, idft(dtft_dict, x))
    plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # #   P L O T   # # # # # # # # # # # # # # # # # # # # # # # # # #
    plot_data = {
        'xs': [x, f, f, f, f],
        'ys': [y, amp, phi, re, im],
        'plot_types': [plt.scatter, plt.stem, plt.stem, plt.stem, plt.stem],
        'subplots': [311, 323, 324, 325, 326],
        'titles': ['Analysed function', 'Amplitude', 'Phase shift', 'Real part', 'Imaginary part'],
        'x_labels': ['x', 'f', 'f', 'f', 'f'],
        'y_labels': ['y(x)', 'A(f)', 'phi(f)', 'Re(f)', 'Im(f)']
    }

    for i in range(len(plot_data['xs'])):
        plt.subplot(plot_data['subplots'][i])
        if i == 0:
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
