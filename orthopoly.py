def legendre(n: int):
    def polynomial(x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return x
        return ((2 * n - 1) * x * legendre(n - 1)(x) - (n - 1) * legendre(n - 2)(x)) / n

    return polynomial


def legendre_prime(n: int):
    def polynomial(x: float) -> float:
        return n * (x * legendre(n)(x) - legendre(n - 1)(x)) / (x**2 - 1)

    return polynomial


def chebyt(n: int):
    def polynomial(x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return x
        return 2 * x * chebyt(n - 1)(x) - chebyt(n - 2)(x)

    return polynomial


def chebyu(n: int):
    def polynomial(x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return 2 * x
        return 2 * x * chebyu(n - 1)(x) - chebyu(n - 2)(x)

    return polynomial
