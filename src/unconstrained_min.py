import numpy as np

class UnconstrainedMin(object):
    WOLFE_CONST = 0.01
    BACKTRACKING = 0.5
    ALPHA = 1

    def __init__(self, obj_tol=1e-8 ,param_tol=1e-12):
        self._obj_tol = obj_tol
        self._param_tol = param_tol

    def check_wolfe(self, f_x, g_x, f_x_next, g_x_next, p):
        cond1 = f_x_next <= f_x + self.BACKTRACKING * self.ALPHA * g_x.T @ p
        cond2 = g_x_next.T @ p >= self.WOLFE_CONST * g_x.T @ p
        return cond1 and cond2

    def line_search_min(self, minimizer, f, x0, max_iter=100):
        iter = 0
        success = False
        x = x0
        record = []
        while not success and iter < max_iter:
            p = minimizer(f, x)
            print(f"Minimizer: p={p}x")
            x_next = x + self.ALPHA * p
            f_x, g_x, _ = f(x, should_hessian=False)
            f_x_next, g_x_next, _ = f(x_next, should_hessian=False)
            success = self.check_wolfe(f_x, g_x, f_x_next, g_x_next, p)
            print(f"Iteration {iter}: x={x}, f(x)={f_x}")
            record.append((x, f_x))
            x = x_next
            iter += 1
        
        return success, x, f_x, record

    def gradient_descent(self, f, x):
        _, gradient, _ = f(x)
        return -1 * gradient

    def newton(self, f, x):
        _, g, h = f(x, should_hessian=True)
        return -1 * np.linalg.inv(h) @ g
        
    # def 