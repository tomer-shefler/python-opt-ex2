import numpy as np

class UnconstrainedMin(object):
    C1 = 0.01
    C2 = 0.5
    ALPHA = 1

    def __init__(self, obj_tol=1e-8 ,param_tol=1e-12):
        self._obj_tol = obj_tol
        self._param_tol = param_tol
        self._b0 = np.eye

    def wolfe_conds(self, f_x, g_x, f_x_next, g_x_next, p, alpha):
        cond1 = f_x_next <= f_x + self.C2 * alpha * g_x.T @ p
        cond2 = g_x_next.T @ p >= self.C2 * g_x.T @ p
        return cond1 and cond2

    def find_next(self, x, f, p):
        """
        Find alpha with wolfe condsm and return x_next and f(x_next)
        """
        alpha = 1
        wolfe_conds_set = False
        f_x, g_x, _ = f(x)
        while not wolfe_conds_set:
            x_next = x + alpha * p
            f_x_next, g_x_next, _ = f(x_next)
            wolfe_conds_set = self.wolfe_conds(f_x, g_x, f_x_next, g_x_next, p, alpha)
            alpha *= 0.5

        return x_next, f_x_next, g_x_next

    def check_tol(self, x, x_next, f_x, f_x_next):
        if np.linalg.norm(x - x_next) < self._param_tol:
            return True
        if np.abs(f_x - f_x_next) < self._obj_tol:
            return True
        return False

    def line_search_min(self, minimizer, f, x0, max_iter=100):
        iter = 0
        success = False
        x = x0
        b = np.eye(len(x))
        record = []
        while not success and iter < max_iter:
            p = minimizer(f, x, b)
            f_x, g_x, _ = f(x, should_hessian=False)
            x_next, f_x_next, g_x_next = self.find_next(x, f, p)
            if minimizer == self.bgfs:
                s = x_next - x
                y = g_x_next - g_x
                b += -1 * (b @ s @ s.T * b) / (s.T @ b @ s)
                b += (y @ y.T) / (y.T @ s)
            success = self.check_tol(x, x_next, f_x, f_x_next)
            print(f"Iteration {iter}: x={x}, f(x)={f_x}")
            record.append((x, f_x))
            x = x_next
            iter += 1
        
        return success, x, f_x, record

    def gradient_descent(self, f, x, *args):
        _, gradient, _ = f(x)
        return -1 * gradient

    def newton(self, f, x, *args):
        _, g, h = f(x, should_hessian=True)
        return -1 * np.linalg.inv(h) @ g
        

    def bgfs(self, f, x, b):
        f_x, g_x, _ = f(x)
        return -b @ g_x
        