import unittest
import numpy
import mpmath
import orthopoly

ns = range(10)
xs = numpy.linspace(-1, 1, 50)


class OrthoPolyTest(unittest.TestCase):
    def test_legendre(self):
        for n in ns:
            for x in xs:
                self.assertAlmostEqual(orthopoly.legendre(n)(x), float(mpmath.legendre(n, x)))

    def test_chebyshev1(self):
        for n in ns:
            for x in xs:
                self.assertAlmostEqual(orthopoly.chebyt(n)(x), float(mpmath.chebyt(n, x)))

    def test_chebyshev2(self):
        for n in ns:
            for x in xs:
                self.assertAlmostEqual(orthopoly.chebyu(n)(x), float(mpmath.chebyu(n, x)))


unittest.main()
