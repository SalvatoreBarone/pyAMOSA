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


class ZDT3(pyamosa.Problem):
    n_var = 30

    def __init__(self):

        pyamosa.Problem.__init__(self, ZDT3.n_var, [pyamosa.Type.REAL] * ZDT3.n_var, [0.0] * ZDT3.n_var, [1.0] * ZDT3.n_var, 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 9 * sum(x[1:]) / (self.num_of_variables - 1)
        h = 1 - np.sqrt(f / g) - (f / g) * np.sin(10* np.pi * f)
        out["f"] = [f, g * h ]

    def optimums(self):
        """
        Optimum:
        0 ≤ 𝑥_1 ≤ 0.0830
        0.1822 ≤ 𝑥_1 ≤ 0.2577
        0.4093 ≤ 𝑥_1 ≤ 0.4538
        0.6183 ≤ 𝑥_1 ≤ 0.6525
        0.8233 ≤𝑥_1 ≤ 0.8518
        𝑥_𝑖 = 0 for 𝑖 = 2,...,𝑛
        """
        out = []
        bounds = [[0, 0.1822, 0.4093, 0.6183, 0.8233], [0.0830, 0.2577, 0.4538, 0.6525, 0.8518]]
        for i in range(len(bounds[0])):
            pareto_set = np.linspace(bounds[0][i], bounds[1][i], 100)
            out = out + [
                        {   "x": [x] + [0] * (ZDT3.n_var-1),
                            "f": [0] * self.num_of_objectives,
                            "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                        } for x in pareto_set
                     ]
        for o in out:
            self.evaluate(o["x"], o)
        return out
