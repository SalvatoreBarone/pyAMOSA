from AMOSA import *

class ZDT3(AMOSA.Problem):
    def __init__(self):
        n_var = 30
        AMOSA.Problem.__init__(self, n_var, [AMOSA.Type.REAL] * n_var, [0] * n_var, [1] * n_var, 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 9 * sum(x[1:]) / (self.num_of_variables - 1)
        h = 1 - np.sqrt(f / g) - (f / g) * np.sin(10* np.pi * f)
        out["f"] = [f, g * h ]
        pass


if __name__ == '__main__':
    problem = ZDT3()
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
    optimizer.save_results(problem, "zdt3.csv")
    optimizer.plot_pareto(problem, "zdt3.pdf")

