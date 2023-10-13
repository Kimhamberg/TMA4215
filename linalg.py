import numpy
from numpy import float64
from numpy.typing import NDArray
import math


def forward_substitution(L: NDArray[float64], b: NDArray[float64]):
    n = len(b)
    x = numpy.zeros(n)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        x[i] = (b[i] - math.fsum(L[i, j] * x[j] for j in range(i))) / L[i, i]
    return x


def backward_substitution(U: NDArray[float64], b: NDArray[float64]):
    n = len(b)
    x = numpy.zeros(n)
    for i in reversed(range(n - 1)):
        x[i] = (b[i] - math.fsum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    return x
