import unittest
import numpy as np
from tests import exmaples
from src import unconstrained_min


class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.u = unconstrained_min.UnconstrainedMin()

    def _test_f(self, f, x0=np.array([1, 1], dtype='int64'), max_iter=100):
        for minimizer in self.u.minimizers:
            self.u.line_search_min(minimizer ,f, x0, max_iter=max_iter)

    # def test_f1(self):
    #     self._test_f(exmaples.f1)

    # def test_f2(self):
    #     self._test_f(exmaples.f2)

    def test_f3(self):
        self._test_f(exmaples.f3)

    def test_rosenbrock(self):
        self._test_f(exmaples.rosenbrock, x0=np.array([-1, 2], dtype='int64'), max_iter=10000)

    def test_e(self):
        self._test_f(exmaples.e_func)

    def test_vect(self):
        self._test_f(exmaples.vect)