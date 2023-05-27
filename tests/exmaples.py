import numpy as np


def gradeint_matrix(Q, x):
    return np.array([
        2 * Q[0][0] * x[0] + x[1] * (Q[0][1]+Q[1][0]),
        x[0] * (Q[0][1]+Q[1][0]) + 2 * Q[1][1] * x[1]
    ])

def hessian_matrix(Q, x):
    return np.array([
        [2*x[0], Q[0][1] + Q[1][0]],
        [Q[0][1] + Q[1][0], 2*x[1]]
    ])

def f1(x, should_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h

def f2(x, should_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h

def f3(x, should_hessian=False):
    Q1 = np.array([[100, 0], [0, 1]])
    Q2 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q = Q2.transpose() @ Q1 @ Q2
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h


from src import unconstrained_min

if __name__ == "__main__":
    u = unconstrained_min.UnconstrainedMin()
    x0 = np.array([1, 1])
    count = 0
    for f in [f2, f3]:
        print(f"f{count} gradient descenet")
        count += 1
        success, x, f_x, _ = u.line_search_min(u.gradient_descent, f, x0)
        print(f"success={success}, x={x}, f(x)={f_x}")
