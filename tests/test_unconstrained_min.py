import unittest
import numpy as np
from tests import examples
from src import unconstrained_min, utils


class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.u = unconstrained_min.UnconstrainedMin()
        self.minimizers = [
            self.u.gradient_descent,
            self.u.newton,
            self.u.bgfs,
            self.u.sr1
        ]

    def _test_f(self, f, x0=np.array([1, 1], dtype='int64'), max_iter=100):
        for minimizer in self.minimizers:
            print(f"test {f.__name__}  {minimizer.__name__}")
            success, x, f_x, record = self.u.line_search_min(minimizer ,f, x0, max_iter=max_iter)
        self.assertTrue(success)

    def test_f1(self):
        self.minimizers.remove(self.u.newton)
        self._test_f(examples.f1)

    def test_f2(self):
        self._test_f(examples.f2)

    def test_f3(self):
        self._test_f(examples.f3)

    def test_rosenbrock(self):
        self.minimizers.remove(self.u.newton)
        self._test_f(examples.rosenbrock, x0=np.array([-1, 2], dtype='int64'), max_iter=10000)

    def test_e(self):
        self._test_f(examples.e_func)

    def test_vect(self):
        self.minimizers.remove(self.u.gradient_descent)
        self.minimizers.remove(self.u.newton)
        self._test_f(examples.vect)