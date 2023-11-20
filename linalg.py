import numpy
from numpy import float64
from numpy.typing import NDArray
import math


def forward_substitution(l: NDArray[float64], b: NDArray[float64]):
    n = len(b)
    x = numpy.zeros(n)
    for i in range(n):
        x[i] = (b[i] - math.fsum(l[i, j] * x[j] for j in range(i))) / l[i, i]
    return x


def backward_substitution(u: NDArray[float64], b: NDArray[float64]):
    n = len(b)
    x = numpy.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - math.fsum(u[i, j] * x[j] for j in range(i + 1, n))) / u[i, i]
    return x


def doolittle(a: NDArray[float64]):
    n = a.shape[0]
    l = numpy.zeros((n, n))
    u = numpy.zeros((n, n))

    for i in range(n):
        l[i, i] = 1
        for j in range(i, n):
            u[i, j] = a[i, j] - math.fsum(l[i, k] * u[k, j] for k in range(i))
            if i != j:
                l[j, i] = (
                    a[j, i] - math.fsum(l[j, k] * u[k, i] for k in range(j))
                ) / u[i, i]
    return l, u


def crout(a: NDArray[float64]):
    n = a.shape[0]
    l = numpy.zeros((n, n))
    u = numpy.zeros((n, n))

    for i in range(n):
        u[i, i] = 1
        for j in range(i, n):
            l[j, i] = a[j, i] - math.fsum(l[j, k] * u[k, i] for k in range(i))
            if i != j:
                u[i, j] = (
                    a[i, j] - math.fsum(l[i, k] * u[k, j] for k in range(i))
                ) / l[i, i]
    return l, u


def cholesky(a: NDArray[float64]):
    n = a.shape[0]
    l = numpy.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                l[i, j] = numpy.sqrt(a[i, i] - numpy.sum(l[i, :] ** 2))
            else:
                l[i, j] = (a[i, j] - numpy.sum(l[i, :] * l[j, :])) / l[j, j]

    return l, l.T


def gram_schmidt_qr(a: NDArray[float64]):
    m, n = a.shape
    q = numpy.zeros((m, n))
    r = numpy.zeros((n, n))

    for j in range(n):
        v = a[:, j]
        for i in range(j):
            r[i, j] = q[:, i] @ a[:, j]
            v = v - r[i, j] * q[:, i]
        r[j, j] = numpy.linalg.norm(v)
        q[:, j] = v / r[j, j]

    return q, r


def householder_qr(a: NDArray[float64]):
    m, n = a.shape
    r = a.copy()
    q = numpy.eye(m)

    for j in range(n):
        x = r[j:, j]
        e1 = numpy.zeros_like(x)
        e1[0] = numpy.linalg.norm(x)
        v = x + numpy.sign(x[0]) * e1
        v = v / numpy.linalg.norm(v)
        h = numpy.eye(m)
        h[j:, j:] -= 2.0 * numpy.outer(v, v)
        r = h @ r
        q = q @ h.T

    return q, r


def givens_rotation(a, b):
    r = numpy.hypot(a, b)
    c = a / r
    s = -b / r

    return c, s


def givens_qr(A: NDArray[float64]):
    m, n = A.shape
    r = A.copy()
    q = numpy.eye(m)

    for j in range(n):
        for i in range(m - 1, j, -1):
            c, s = givens_rotation(r[i - 1, j], r[i, j])
            g = numpy.eye(m)
            g[i - 1 : i + 1, i - 1 : i + 1] = numpy.array([[c, s], [-s, c]])
            r = g @ r
            q = q @ g.T

    return q, r


def jacobi(
    a: NDArray[float64], b: NDArray[float64], tolerance=1e-10, max_iterations=1000
):
    x = numpy.zeros_like(b, dtype=numpy.float64)
    d = numpy.diag(a)
    r = a - numpy.diagflat(d)

    for _ in range(max_iterations):
        x_new = (b - r @ x) / d
        if numpy.linalg.norm(x_new - x, ord=numpy.inf) < tolerance:
            return x_new
        x = x_new

    return x


def gauss_seidel(
    a: NDArray[float64], b: NDArray[float64], tolerance=1e-10, max_iterations=1000
):
    x = numpy.zeros_like(b, dtype=numpy.float64)
    for _ in range(max_iterations):
        x_new = numpy.copy(x)
        for i in range(a.shape[0]):
            sum1 = numpy.dot(a[i, :i], x_new[:i])
            sum2 = numpy.dot(a[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - sum1 - sum2) / a[i, i]
        if numpy.linalg.norm(x_new - x, ord=numpy.inf) < tolerance:
            return x_new
        x = x_new
    return x


def jacobi_relaxed(
    a: NDArray[float64],
    b: NDArray[float64],
    omega: float,
    tolerance=1e-10,
    max_iterations=1000,
):
    x = numpy.zeros_like(b, dtype=numpy.float64)
    d = numpy.diag(a)
    r = a - numpy.diagflat(d)
    for _ in range(max_iterations):
        x_jacobi = (b - r @ x) / d
        x_new = omega * x_jacobi + (1 - omega) * x
        if numpy.linalg.norm(x_new - x, ord=numpy.inf) < tolerance:
            return x_new
        x = x_new

    return x


def gauss_seidel_relaxed(
    a: NDArray[float64],
    b: NDArray[float64],
    omega: float,
    tolerance=1e-10,
    max_iterations=1000,
):
    x = numpy.zeros_like(b, dtype=numpy.double)
    for _ in range(max_iterations):
        x_new = numpy.copy(x)
        for i in range(a.shape[0]):
            sum1 = numpy.dot(a[i, :i], x_new[:i])
            sum2 = numpy.dot(a[i, i + 1 :], x[i + 1 :])
            x_new[i] = x[i] + omega * (b[i] - sum1 - sum2) / a[i, i]
        if numpy.linalg.norm(x_new - x, ord=numpy.inf) < tolerance:
            return x_new
        x = x_new
    return x


def doolittle(A: NDArray[float64]):
    n = A.shape[0]
    L = numpy.zeros((n, n))
    U = numpy.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - math.fsum(L[i, k] * U[k, j] for k in range(i))
            if i != j:
                L[j, i] = (
                    A[j, i] - math.fsum(L[j, k] * U[k, i] for k in range(j))
                ) / U[i, i]
    return L, U


def crout(A: NDArray[float64]):
    n = A.shape[0]
    L = numpy.zeros((n, n))
    U = numpy.zeros((n, n))

    for i in range(n):
        U[i, i] = 1
        for j in range(i, n):
            L[j, i] = A[j, i] - math.fsum(L[j, k] * U[k, i] for k in range(i))
            if i != j:
                U[i, j] = (
                    A[i, j] - math.fsum(L[i, k] * U[k, j] for k in range(i))
                ) / L[i, i]
    return L, U


def cholesky(A: NDArray[float64]):
    n = A.shape[0]
    L = numpy.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = numpy.sqrt(A[i, i] - numpy.sum(L[i, :] ** 2))
            else:
                L[i, j] = (A[i, j] - numpy.sum(L[i, :] * L[j, :])) / L[j, j]
    return L, L.T


def gram_schmidt_qr(A: NDArray[float64]):
    m, n = A.shape
    Q = numpy.zeros((m, n))
    R = numpy.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = numpy.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def householder_qr(A: NDArray[float64]):
    m, n = A.shape
    R = A.copy()
    Q = numpy.eye(m)

    for j in range(n):
        x = R[j:, j]
        e1 = numpy.zeros_like(x)
        e1[0] = numpy.linalg.norm(x)
        v = x + numpy.sign(x[0]) * e1
        v = v / numpy.linalg.norm(v)

        # Householder transformation
        H = numpy.eye(m)
        H[j:, j:] -= 2.0 * numpy.outer(v, v)
        R = H @ R
        Q = Q @ H.T

    return Q, R


def givens_rotation(a, b):
    r = numpy.hypot(a, b)
    c = a / r
    s = -b / r
    return c, s


def givens_qr(A):
    m, n = A.shape
    R = A.copy()
    Q = numpy.eye(m)

    for j in range(n):
        for i in range(m - 1, j, -1):
            c, s = givens_rotation(R[i - 1, j], R[i, j])

            # Apply Givens rotation
            G = numpy.eye(m)
            G[i - 1 : i + 1, i - 1 : i + 1] = numpy.array([[c, s], [-s, c]])
            R = G @ R
            Q = Q @ G.T

    return Q, R
