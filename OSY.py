from AMOSA import *

class OSY(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 6, [AMOSA.Type.REAL] * 6, [0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10], 2, 6)

    def evaluate(self, x, out):
        f1 = -(25 * (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 + (x[3] - 4) ** 2 + (x[4] - 1) ** 2 )
        f2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2
        g1 = 2 - x[0] - x[1]
        g2 = x[0] + x[1] -6
        g3 = x[1] - x[0] - 2
        g4 = x[0] - 3 * x[1] - 2
        g5 = x[3]  + (x[2] - 3) ** 2 - 4
        g6 = 4 - x[5] - (x[4] - 3) ** 2
        out["f"] = [f1, f2]
        out["g"] = [g1, g2, g2, g3, g4, g5, g6]

if __name__ == '__main__':
    problem = OSY()
    optimizer = AMOSA()
    optimizer.archive_hard_limit = 50
    optimizer.archive_soft_limit = 100
    optimizer.initial_refinement_iterations = 2500
    optimizer.archive_gamma = 2
    optimizer.refinement_iterations = 2500
    optimizer.initial_temperature = 500
    optimizer.final_temperature = 0.0000001
    optimizer.cooling_factor = 0.9
    optimizer.minimize(problem)
    optimizer.save_results(problem, "osy.csv")
    optimizer.plot_pareto(problem, "osy.pdf")

