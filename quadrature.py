import math
import polyroot
import orthopoly
from typing import Callable


# https://en.wikipedia.org/wiki/Gauss-Legendre_quadrature
def gauss_legendre(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    zeros: list[float] = list()
    weights: list[float] = list()
    k = 1
    while 1 <= k <= n:
        # use good starting points: https://en.wikipedia.org/wiki/Chebyshev_nodes
        node = math.cos(((2 * k - 1) / 2 * n) * math.pi)
        # find zero of n-degree legendre and append: https://en.wikipedia.org/wiki/Newton%27s_method
        zero = polyroot.NewtonRaphson(
            orthopoly.legendre(n), orthopoly.legendre_prime(n), node
        )
        zeros.append(zero)
        # calculate associated weight and append
        weight = 2 / ((1 - zero**2) * orthopoly.legendre_prime(n)(zero) ** 2)
        weights.append(weight)
        k += 1
    # change interval: https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
    zeros = [(b - a) / 2 * zero + (a + b) / 2 for zero in zeros]
    weights = [(b - a) / 2 * weight for weight in weights]
    integral = math.fsum(weight * f(zero) for weight, zero in zip(zeros, weights))
    return integral
