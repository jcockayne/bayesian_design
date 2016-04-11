import unittest
import numpy as np
from bayesian_design.ace import a_optimality

class SympyHelpersTest(unittest.TestCase):
    def test_2d(self):
        cov = np.array([[1., 5.], [2., 4.]])
        tmp = a_optimality(lambda x: cov)(None)
        np.testing.assert_almost_equal(tmp, 5.)

    def test_1d(self):
        cov = np.array([1., 4.])
        tmp = a_optimality(lambda x: cov)(None)
        np.testing.assert_almost_equal(tmp, 5.)

        cov = cov.reshape((2, 1))
        tmp = a_optimality(lambda x: cov)(None)
        np.testing.assert_almost_equal(tmp, 5.)

        cov = cov.T
        tmp = a_optimality(lambda x: cov)(None)
        np.testing.assert_almost_equal(tmp, 5.)
