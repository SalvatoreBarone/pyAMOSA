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
import pyamosa, numpy as np


class DTZ1(pyamosa.Problem):

    def __init__(self, n_vars, n_objs):
        self.k = n_vars - n_objs + 1
        pyamosa.Problem.__init__(self, n_vars, [pyamosa.Type.REAL] * n_vars, [0.0] * n_vars, [1.0] * n_vars, n_objs, 0)

    def evaluate(self, x, out):
        f = np.zeros((self.num_of_objectives,))
        g = 100 * ( self.k + np.sum( (x[-self.k:] - 0.5) **2 - np.cos(20 * np.pi * (x[-self.k:] - 0.5)) ) )

        f[0] = 0.5 * (1 + g) * np.prof(x[:self.num_of_objectives-2])

        for m in range(1, self.num_of_objectives-1):
            f[m] = 0.5 * (1 + g) * (1 - x[self.num_of_objectives-m]) * np.prod(x[:self.num_of_objectives-m-1])
            
        f[self.num_of_objectives-1] = 0.5 * (1 + g) * (1 - x[0])
        out["f"] = f.tolist()

    def optimums(self):
        pareto = []
        for _ in range(100):
            c = {"x": [0] * self.num_of_variables, "f": [0] * self.num_of_objectives, "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None}
        
            self.evaluate(c["x"], c)
            if np.sum(c["f"][:self.num_of_objectives-1]) - 0.5 < np.finfo(float).eps:
                pareto.append(c)
        return pareto

