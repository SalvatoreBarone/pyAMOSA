from AMOSA import *

class BNH(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 2, [AMOSA.Type.REAL] * 2, [0]*2, [5, 3], 2, 2)

    def evaluate(self, x, out):
        f1 = 4 * x[0] ** 2 + 4 * x[1] ** 2
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        g1 = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        g2 = 7.7 - (x[0] - 5) ** 2 - (x[1] + 3) ** 2
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]

if __name__ == '__main__':
    problem = BNH()
    optimizer = AMOSA()
    optimizer.archive_hard_limit = 50
    optimizer.archive_soft_limit = 150
    optimizer.initial_refinement_iterations = 2500
    optimizer.archive_gamma = 2
    optimizer.refinement_iterations = 2500
    optimizer.initial_temperature = 500
    optimizer.final_temperature = 0.0000001
    optimizer.cooling_factor = 0.8
    optimizer.minimize(problem)
    optimizer.save_results(problem, "bnh.csv")
    optimizer.plot_pareto(problem, "bnh.pdf")

