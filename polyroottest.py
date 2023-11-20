import polyroot
import unittest
from typing import Callable
import math

p2: Callable[[float], float] = lambda x: (3 * x**2 - 1) / 2
p2_prime: Callable[[float], float] = lambda x: 3 * x
p3: Callable[[float], float] = lambda x: (5 * x**3 - 3 * x) / 2
p3_prime: Callable[[float], float] = lambda x: (15 * x**2 - 3) / 2
p4: Callable[[float], float] = lambda x: (35 * x**4 - 30 * x**2 + 3) / 8
p4_prime: Callable[[float], float] = lambda x: (140 * x**3 - 60 * x) / 8

<<<<<<< HEAD

=======
>>>>>>> 65d4086dc8457f8d38657a7b594010841a2547ea
class PolyrootTest(unittest.TestCase):
    def test_legendre(self):
        polynomial_roots = (
            (-1 / math.sqrt(3), 1 / math.sqrt(3)),
            (-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)),
            (
                -math.sqrt(3 / 7 - (2 * math.sqrt(6 / 5) / 7)),
                +math.sqrt(3 / 7 - (2 * math.sqrt(6 / 5) / 7)),
                -math.sqrt(3 / 7 + (2 * math.sqrt(6 / 5) / 7)),
                +math.sqrt(3 / 7 + (2 * math.sqrt(6 / 5) / 7)),
            ),
        )
        computed_roots = (
            polyroot.NewtonRaphson(p2, p2_prime, 1),
            polyroot.NewtonRaphson(p3, p3_prime, 1),
            polyroot.NewtonRaphson(p4, p4_prime, 1),
        )
        for expected_roots, computed_root in zip(polynomial_roots, computed_roots):
            for expected_root in expected_roots:
                if round(abs(computed_root - expected_root), 7) == 0:
                    return
            self.fail(
                f"{computed_root} not almost equal to any root in {expected_roots}"
            )


unittest.main()
