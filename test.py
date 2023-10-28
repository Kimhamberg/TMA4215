n = 10
import numpy
import math

x = numpy.arange(1, 7).reshape(2, 3)

print(numpy.sum(x))
print(math.fsum(x.flat))