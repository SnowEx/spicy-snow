import numpy

def to01(array):
    a = array.min()
    # ignore the Runtime Warning
    with numpy.errstate(divide='ignore'):
        b = 1. /(array.max() - array.min())
    if not(numpy.isfinite(b)):
        b = 0
    return numpy.vectorize(lambda x: b * (x - a))(array)