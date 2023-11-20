import numpy
from numpy import float64
from numpy.typing import NDArray


# Cooley–Tukey radix-2 FFT: https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm
def fft(x: NDArray[float64], n: int) -> NDArray[float64] | ValueError:
    if n & (n - 1) != 0 or n == 0:
        raise ValueError("n must be a power of 2.")
    if n == 1:
        return x

    omega = numpy.exp(2 * numpy.pi * 1j / n)
    nhalf = n // 2
    even = fft(x[0::2], nhalf)
    odd = fft(x[1::2], nhalf)
    index = numpy.arange(nhalf)
    factor = omega**index
    combined = numpy.concatenate([even + factor * odd, even - factor * odd])
    return combined
