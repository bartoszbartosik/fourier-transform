# Discrete Fourier Transform
The Fourier Transform is a mathematical tool which is used to express given function in a domain of frequencies, 
which this function consists of. In other words, the idea is to assume, that the function which is the object of the 
analysis, is a sum of sinusoids of different amplitudes $A$, frequencies $f$ and phase shifts $\phi$ - the Fourier Transform decomposes this function into separate sinusoids. It is described as follows:
```math
Y(f) = \int_{-\infty}^{\infty} y(x) \cdot e^{-2\pi fix} \,dx 
```
The output of this operation is another function of complex values.

The program works with data stored in a form of samples, i.e. discrete representation. This application yields the change from the continuous Fourier Transform form to its discretized equivalent, Discrete Fourier Transform:
```math
X_{k} = \sum_{n=0}^{N-1} x_n \cdot e^{-\frac{i2\pi}{N}kn}
```
where:
- $k$: k-th argument of the transformed function;
- $X_k$: k-th value of the transformed function;
- $n$: n-th argument of the original function;
- $x_n$: n-th value of the original function;
- $N$: number of samples.

## Implementation
In order to fetch the real and imaginary part of a result of above summation, it has been rearranged using complex numbers properties. Knowing, that:
```math
re^{i\phi} = r\cos{\phi} + ri\sin{\phi}
```
it can be written, that:
```math
X_{k} = \sum_{n=0}^{N-1} x_n \cdot \cos{\frac{2\pi}{N}kn} - x_n \cdot \sin{\frac{2\pi}{N}kn}
```
where:
```math
x_n \cdot \cos{\frac{2\pi}{N}kn} = Re(X_k)
```
```math
- x_n \cdot \sin{\frac{2\pi}{N}kn} = Im(X_k)
```

## Frequency representation
Having Fourier Transform results, the original function can be represented in its frequencies domain. The amplitude $A$ and phase shift $\phi$ for each frequency $k$ can be extracted using the expressions below:
```math
A_k = \frac{1}{N}\sqrt{Re(X_k)^2 + Im(X_k)^2}
```
```math
\phi_k = atan2(Re(X_k), Im(X_k))
```

# Inverse Fourier Transform
In order to go back from the frequency representation to the original form of the function, an Inverse Fourier Transform has to be performed. This is done by the following expression:
```math
x_{n} = \frac{1}{N}\sum_{k=0}^{N-1} X_k \cdot e^{\frac{i2\pi}{N}kn}
```

## Implementation
Using the same complex numbers properties, the equation above can be expressed as following:
```math
x_{n} = \frac{1}{N}\sum_{k=0}^{N-1} X_k (\cos{\frac{2\pi}{N}kn} + \sin{\frac{2\pi}{N}kn})
```
where:
```math
X_k = Re(X_k) + Im(X_k)i
```
using the multiplication rule for complex numbers $(x+yi)(u+vi)=(xu-yv)+(xv+yu)i$, the Inverse Fourier Transform can be written as:
```math
x_{n} = \frac{1}{N}\sum_{k=0}^{N-1} (Re(X_k)\cos{\frac{2\pi}{N}kn} + Im(X_k)\sin{\frac{2\pi}{N}kn}) + (Re(X_k)\sin{\frac{2\pi}{N}kn} - Im(X_k)\cos{\frac{2\pi}{N}kn})i
```

# Visualization


