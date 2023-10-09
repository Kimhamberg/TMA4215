from typing import Callable
def NewtonRaphson(f: Callable[[float], float], f_prime: Callable[[float], float], x: float):
    old_x, counter, maxiterations = x, 0, 1000
    while(counter < maxiterations):
        new_x = old_x - f(old_x)/f_prime(old_x)
        if (abs(new_x - old_x) < 1e-6):
            return new_x
        old_x, counter = new_x, counter + 1
    return 0.0
