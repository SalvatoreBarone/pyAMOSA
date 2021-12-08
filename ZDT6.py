import numpy as np

from AMOSA import *

class ZDT6(AMOSA.Problem):
    def __init__(self):
        n_var = 10
        AMOSA.Problem.__init__(self, n_var, [AMOSA.Type.REAL] * n_var, [0]*n_var, [1] * n_var, 2, 0)

    def evaluate(self, x, out):
        f = 1 - np.exp(-4 * x[0]) * np.power(np.sin(6 * np.pi * x[0]), 6)
        g = 1 + 9 * np.power(sum(x[1:]) / 9, 1./4)
        h = 1 - (f / g) ** 2
        out["f"] = [f, g * h ]
        pass


if __name__ == '__main__':
    problem = ZDT6()
    optimizer = AMOSA()
    optimizer.archive_hard_limit = 50
    optimizer.archive_soft_limit = 150
    optimizer.initial_refinement_iterations = 1500
    optimizer.archive_gamma = 2
    optimizer.refinement_iterations = 2500
    optimizer.initial_temperature = 500
    optimizer.final_temperature = 0.0000001
    optimizer.cooling_factor = 0.8
    optimizer.minimize(problem)
    optimizer.save_results(problem, "zdt6.csv")
    optimizer.plot_pareto(problem, "zdt6.pdf")

