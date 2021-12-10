from AMOSA import *

class TNK(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 2, [AMOSA.Type.REAL] * 2, [0.0000001] * 2, [np.pi] * 2, 2, 2)

    def evaluate(self, x, out):
        f1 = x[0]
        f2 = x[1]
        g1 = 1 + 0.1 * np.cos( 16 * np.arctan(x[0] / x[1])) - x[0] ** 2 - x[1] ** 2
        g2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.5
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]

if __name__ == '__main__':
    problem = TNK()
    optimizer = AMOSA()
    optimizer.archive_hard_limit = 50
    optimizer.archive_soft_limit = 100
    optimizer.initial_refinement_iterations = 2500
    optimizer.archive_gamma = 3
    optimizer.refinement_iterations = 2500
    optimizer.initial_temperature = 500
    optimizer.final_temperature = 0.0000001
    optimizer.cooling_factor = 0.8
    optimizer.minimize(problem)
    optimizer.save_results(problem, "tnk.csv")
    optimizer.plot_pareto(problem, "tnk.pdf")

