import cmath

# Cooley–Tukey radix-2 FFT: https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm
def fft(x: list[complex], n: int) -> list[complex]:
    if n == 1:
        return x
    omega = cmath.exp(cmath.tau*1j/n)
    nhalf = n//2
    even = fft(x[0::2], nhalf)
    odd = fft(x[1::2], nhalf)
    left = [even[i] + odd[i]*omega**i for i in range(nhalf)]
    right = [even[i] - odd[i]*omega**i for i in range(nhalf)]
    return left + right
