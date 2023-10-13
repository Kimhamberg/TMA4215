import unittest
import numpy
import linalg


class TestForwardSubstitution(unittest.TestCase):
    def test_forward_sub(self):
        # Create a random lower triangular matrix and vector b
        n = numpy.random.randint(2, 16)
        L = numpy.tril(numpy.random.rand(n, n))
        b = numpy.random.rand(n)

        # Use our forward substitution method
        x_custom = linalg.forward_substitution(L, b)

        # Use numpy's built-in solve function
        x_numpy = numpy.linalg.solve(L, b)

        # Assert that the two solutions are almost equal
        numpy.testing.assert_almost_equal(x_custom, x_numpy)

    def test_backward_sub(self):
        # Create a random upper triangular matrix and vector b
        n = numpy.random.randint(2, 16)
        U = numpy.triu(numpy.random.rand(n, n))
        b = numpy.random.rand(n)

        # Use our backward substitution method
        x_custom = linalg.backward_substitution(U, b)

        # Use numpy's built-in solve function
        x_numpy = numpy.linalg.solve(U, b)

        # Assert that the two solutions are almost equal
        numpy.testing.assert_almost_equal(x_custom, x_numpy)


unittest.main()
