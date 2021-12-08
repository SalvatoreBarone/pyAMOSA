from AMOSA import *

class ZDT4(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 10, [AMOSA.Type.REAL] * 10, [0, -10, -10, -10, -10, -10, -10, -10, -10, -10], [1, 10, 10, 10, 10, 10, 10, 10, 10, 10], 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 10 * 9 + sum( [ i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:] ] )
        h = 1 - np.sqrt(f / g)
        out["f"] = [f, g * h ]
        pass


if __name__ == '__main__':
    problem = ZDT4()
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
    optimizer.save_results(problem, "zdt4.csv")
    optimizer.plot_pareto(problem, "zdt4.pdf")

