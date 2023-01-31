"""
Copyright 2021-2022 Salvatore Barone <salvatore.barone@unina.it>

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


class BNH(pyamosa.Problem):
    n_var = 2

    def __init__(self):
        pyamosa.Problem.__init__(self, BNH.n_var, [pyamosa.Type.REAL] * BNH.n_var, [0.0] * BNH.n_var, [5.0, 3.0], 2, 2)

    def evaluate(self, x, out):
        f1 = 4 * x[0] ** 2 + 4 * x[1] ** 2
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        g1 = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        g2 = 7.7 - (x[0] - 5) ** 2 - (x[1] + 3) ** 2
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]

    def optimums(self):
        pareto_set = np.linspace(0, 3, 100)
        out = [{    "x": [x, x],
                    "f": [0] * self.num_of_objectives,
                    "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None} for x in pareto_set ]
        pareto_set = np.linspace(3, 5, 100)
        out += [{  "x": [x, 3],
                   "f": [0] * self.num_of_objectives,
                   "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None} for x in pareto_set ]
        for o in out:
            self.evaluate(o["x"], o)
        return out
