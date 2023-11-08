# Discrete Fourier Transform
## Introduction
The Fourier Transform is a mathematical tool which is used to express given function in a domain of frequencies, 
which this function consists of. In other words, the idea is to assume, that the function which is the object of the 
analysis, is a sum of sinusoids of different amplitudes $A$, frequencies $f$ and phase shifts $\phi$ - the Fourier Transform decomposes this function into separate sinusoids. It is described as follows:
```math
F(f) = \int_{-\infty}^{\infty} f(x) \cdot e^{-2\pi fix} \,dx 
```
The output of this operation is another function of complex values.

The program works with data stored in a form of samples, i.e. discrete representation. This allows for modification of the continuous Fourier Transform formula into its discretized equivalent Discrete Fourier Transform, which can be expressed as follows:
```math
F(f) = \int_{-\infty}^{\infty} f(x) \cdot e^{-2\pi fix} \,dx 
```
