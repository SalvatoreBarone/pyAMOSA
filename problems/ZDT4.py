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


class ZDT4(pyamosa.Problem):
    n_var = 10
    def __init__(self):
        pyamosa.Problem.__init__(self, ZDT4.n_var, [pyamosa.Type.REAL] * ZDT4.n_var, [0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 10 * 9 + sum(i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:])
        h = 1 - np.sqrt(f / g)
        out["f"] = [f, g * h ]

    def optimums(self):
        """
        Optimum:
        0 <= x_1 <= 1, x_i = 0 for each i in 2...n
        """
        pareto_set = np.linspace(0, 1, 100)
        out =   [
                    {   "x": [x] + [0] * (ZDT4.n_var-1),
                        "f": [0] * self.num_of_objectives,
                        "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                    } for x in pareto_set
                 ]
        for o in out:
            self.evaluate(o["x"], o)
        return out
