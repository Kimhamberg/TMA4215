import math
import polyroot
import orthopoly


def gauss_legendre(n: int) -> tuple[float, float]:
    for k in range(1, n + 1):
        x = math.cos(((2 * k - 1) / 2 * n) * math.pi)
        polyroot.NewtonRaphson(orthopoly.legendre, orthopoly.legendre_prime, x)
        

    return (2.0, 3.5)
