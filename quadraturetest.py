import unittest
import quadrature


class GaussLegendreTest(unittest.TestCase):
    def test_parabola(self):
        def parabola(x: float) -> float:
            return x**3

        START = 3.0
        END = 5.0
        SAMPLE_SIZE = 2
        # https://www.integral-calculator.com/
        self.assertAlmostEqual(
            quadrature.gauss_legendre(parabola, START, END, SAMPLE_SIZE), 16.0
        )


unittest.main()
