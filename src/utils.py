import matplotlib.pyplot as plt
import numpy as np



def plot_contour_function(objective_func):
    
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    z = np.zeros(([len(x), len(y)]))

    for i in range(0, len(x)):
        for j in range(0, len(y)):
            z[j, i], _, _  = objective_func(np.array([x[i], y[j]]))

    plt.figure()
    plt.contourf(x, y, z, 20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(objective_func.__name__)
    plt.colorbar()
    plt.show()

def plot_function_values(records):
    plt.figure()
    for method_name, values in iterations.items():
        plt.plot(range(len(values)), values, label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()

from tests import examples

if __name__ == "__main__":
    func_list = [examples.f1, examples.f2, examples.f3,
        examples.rosenbrock, examples.vect, examples.e_func]
    for f in func_list:
        plot_contour_function(f)
