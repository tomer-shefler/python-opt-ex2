import matplotlib.pyplot as plt
import numpy as np

def plot_contour_function(objective_func, xlim, ylim, title):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = objective_func(X, Y)

    plt.figure()
    plt.contour(X, Y, Z, levels=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_function_values(iterations, method_names):
    plt.figure()
    for method_name, values in iterations.items():
        plt.plot(range(len(values)), values, label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()
