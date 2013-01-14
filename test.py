import numpy as np
from scipy import linalg
from numpy.testing import assert_almost_equal

from trace import trace

def test_1():
    B = np.ones((3, 3))
    X = np.random.randn(100, 9)
    y = np.dot(X, B.ravel('F')) + .1 * np.random.randn(100)

    B_, _ = trace(X, y, 10., 0., (3, 3), rtol=1e-10)
    s = linalg.svdvals(B_)
    assert np.sum(s > 1e-6) == 1

    assert_almost_equal(B, B_, decimal=1)


