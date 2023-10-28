import numpy
from numpy import float64
from numpy.typing import NDArray
import math


def forward_substitution(L: NDArray[float64], y: NDArray[float64]):
    n = len(y)
    x = numpy.zeros(n)
    for i in range(n):
        x[i] = (y[i] - math.fsum(L[i, j] * x[j] for j in range(i)))/L[i, i]
    return x


def backward_substitution(U: NDArray[float64], y: NDArray[float64]):
    n = len(y)
    x = numpy.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - math.fsum(U[i, j]*x[j] for j in range(i+1, n)))/U[i, i]
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
                L[j, i] = (A[j, i] - math.fsum(L[j, k] * U[k, i] for k in range(j))) / U[i, i]
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
                U[i, j] = (A[i, j] - math.fsum(L[i, k] * U[k, j] for k in range(i))) / L[i, i]
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
        for i in range(m-1, j, -1):
            c, s = givens_rotation(R[i-1, j], R[i, j])
            
            # Apply Givens rotation
            G = numpy.eye(m)
            G[i-1:i+1, i-1:i+1] = numpy.array([[c, s], [-s, c]])
            R = G @ R
            Q = Q @ G.T
            
    return Q, R