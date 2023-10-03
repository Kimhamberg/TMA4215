def legendre(n: int, x: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return ((2 * n - 1) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n


def chebyt(n: int, x: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2 * x * chebyt(n - 1, x) - chebyt(n - 2, x)


def chebyu(n: int, x: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return 2 * x
    return 2 * x * chebyu(n - 1, x) - chebyu(n - 2, x)
