"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import pyamosa, numpy as np, itertools

"""
For further details, please refer to

    Huband, S., P. Hingston, L. Barone, e L. While. "A review of multiobjective test problems and a scalable test problem toolkit".
    IEEE Transactions on Evolutionary Computation 10, fasc. 5 (ottobre 2006): 477-506. https://doi.org/10.1109/TEVC.2005.861417.

    Cheng, Ran, Yaochu Jin, Markus Olhofer, e Bernhard sendhoff. "Test Problems for Large-Scale Multiobjective and Many-Objective
    Optimization". IEEE Transactions on Cybernetics 47, fasc. 12 (dicembre 2017): 4108-21. https://doi.org/10.1109/TCYB.2016.2600577.

    Deb, Kalyanmoy, Lothar Thiele, Marco Laumanns, e Eckart Zitzler. "Scalable Test Problems for Evolutionary Multiobjective 
    Optimization". In Evolutionary Multiobjective Optimization, a cura di Ajith Abraham, Lakhmi Jain, e Robert Goldberg, 105-45. 
    Advanced Information and Knowledge Processing. London: Springer-Verlag, 2005. https://doi.org/10.1007/1-84628-137-7_6.

    Zitzler, Eckart, Kalyanmoy Deb, e Lothar Thiele. "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results"
    https://doi.org/10.3929/ETHZ-A-004287264.

"""

class DTZ2(pyamosa.Problem):

    def __init__(self, n_vars, n_objs):
        self.k = n_vars - n_objs + 1
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.REAL] * n_vars, [0.0] * n_vars, [1.0] * n_vars, n_objs, 0)

    def evaluate(self, x, out):
        f = np.zeros((self.num_of_objectives,))
        g = np.sum( np.square(x[self.num_of_objectives-1:] - 0.5) )
        f[0] = 0.5 * (1 + g) * np.prod( np.cos(np.pi * 0.5* x[:self.num_of_objectives-2]))

        for m in range(1, self.num_of_objectives-1):
            f[m] = (1 + g) * np.sin(np.pi * 0.5 * x[self.num_of_objectives-m]) * np.prod( np.cos(np.pi * 0.5 * x[:self.num_of_objectives-m-1]))

        f[self.num_of_objectives-1] = (1 + g) * np.sin(np.pi * 0.5 * x[0])
        out["f"] = f.tolist()

    def optimums(self):
        pareto_front = []
        candidate_pareto_set = [ np.arange(self.lower_bound[i], self.upper_bound[i], 0.01).tolist() for i in range(self.num_of_objectives) ]
        for element in itertools.product(*candidate_pareto_set):
            c = {"x": element + [0] * self.self.k, "f": [0] * self.num_of_objectives, "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None}
            self.evaluate(c["x"], c)
            if np.sum(c["f"][:self.num_of_objectives-1] ** 2) - 1 < np.finfo(float).eps:
                pareto_front.append(c)
        return pareto_front

