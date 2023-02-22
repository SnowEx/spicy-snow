import numpy

import logging
log = logging.getLogger(__name__)

def to01(array):

    log.debug("Making {array} to 0-1 range.")

    a = array.min()
    # ignore the Runtime Warning

    with numpy.errstate(divide='ignore'):
        b = 1. /(array.max() - array.min())

    if not(numpy.isfinite(b)):
        b = 0
        
    return numpy.vectorize(lambda x: b * (x - a))(array)