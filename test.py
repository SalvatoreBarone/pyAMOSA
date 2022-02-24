import numpy as np

from AMOSA import *

class Test(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 2, [AMOSA.Type.INTEGER] * 2, [0] * 2, [100] * 2, 2, 0)

    def evaluate(self, x, out):
        out["f"] = x
        pass


if __name__ == '__main__':
    problem = Test()
    optimizer = AMOSA()
    optimizer.archive_hard_limit = 5
    optimizer.archive_soft_limit = 10
    optimizer.initial_refinement_iterations = 2
    optimizer.archive_gamma = 2
    optimizer.refinement_iterations = 2
    optimizer.initial_temperature = 50
    optimizer.final_temperature = 0.0001
    optimizer.cooling_factor = 0.9
    optimizer.early_termination = 15
    optimizer.minimize(problem)
    optimizer.save_results(problem, "test.csv")
    optimizer.plot_pareto(problem, "test.pdf")

