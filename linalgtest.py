import unittest
import numpy
import linalg


class LinalgTest(unittest.TestCase):
    def test_forward_sub(self):
        n = numpy.random.randint(2, 16)
        L = numpy.tril(numpy.random.rand(n, n))
        b = numpy.random.rand(n)
        x_custom = linalg.forward_substitution(L, b)
        x_numpy = numpy.linalg.solve(L, b)
        numpy.testing.assert_allclose(x_custom, x_numpy)

    def test_backward_sub(self):
        n = numpy.random.randint(2, 16)
        U = numpy.triu(numpy.random.rand(n, n))
        b = numpy.random.rand(n)
        x_custom = linalg.backward_substitution(U, b)
        x_numpy = numpy.linalg.solve(U, b)
        numpy.testing.assert_allclose(x_custom, x_numpy)

    def test_doolittle(self):
        n = numpy.random.randint(2, 16)
        A = numpy.random.rand(n, n)
        L, U = linalg.doolittle(A)
        numpy.testing.assert_allclose(L, numpy.tril(L))
        numpy.testing.assert_allclose(U, numpy.triu(U))
        numpy.testing.assert_allclose(L @ U, A)

    def test_crout(self):
        n = numpy.random.randint(2, 16)
        A = numpy.random.rand(n, n)
        L, U = linalg.crout(A)
        numpy.testing.assert_allclose(L, numpy.tril(L))
        numpy.testing.assert_allclose(U, numpy.triu(U))
        numpy.testing.assert_allclose(L @ U, A)

    def test_cholesky(self):
        n = numpy.random.randint(2, 16)
        B = numpy.random.rand(n, n)
        A = B.T @ B
        L, U = linalg.cholesky(A)
        numpy.testing.assert_allclose(L, numpy.tril(L))
        numpy.testing.assert_allclose(U, numpy.triu(U))
        numpy.testing.assert_allclose(L @ U, A)

    def test_jacobi(self):
        A = numpy.asarray(
            [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]]
        )
        b = numpy.asarray([6, 25, -11, 15])
        solution = linalg.jacobi(A, b)
        numpy.testing.assert_array_almost_equal(numpy.dot(A, solution), b)

    def test_gauss_seidel(self):
        A = numpy.asarray(
            [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]]
        )
        b = numpy.asarray([6, 25, -11, 15])
        solution = linalg.gauss_seidel(A, b)
        numpy.testing.assert_array_almost_equal(numpy.dot(A, solution), b)


unittest.main()
