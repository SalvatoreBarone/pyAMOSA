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


class OSY(pyamosa.Problem):
    def __init__(self):
        pyamosa.Problem.__init__(self, 6, [pyamosa.Type.REAL] * 6, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [10.0, 10.0, 5.0, 6.0, 5.0, 10.0], 2, 6)

    def evaluate(self, x, out):
        f1 = -(25 * (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 + (x[3] - 4) ** 2 + (x[4] - 1) ** 2 )
        f2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2
        g1 = 2 - x[0] - x[1]
        g2 = x[0] + x[1] - 6
        g3 = x[1] - x[0] - 2
        g4 = x[0] - 3 * x[1] - 2
        g5 = x[3]  + (x[2] - 3) ** 2 - 4
        g6 = 4 - x[5] - (x[4] - 3) ** 2
        out["f"] = [f1, f2]
        out["g"] = [g1, g2, g2, g3, g4, g5, g6]

    def optimums(self):
        """
        The Pareto-optimal region is a concatenation of five regions. Every region lies on some of the constraints. However, for the entire Pareto-optimal region,
        𝑥_4=𝑥_6=0. In table below shows the other variable values in each of the five regions and the constraints that are active in each region.
        x_1             x_2             x_3             x_5
        5               1               (1,5)           5
        5               1               (1,5)           1
        (4.056, 5)      (x_1-2)/3       1               1
        0               2               (1, 3.732)      1
        (0, 1)          2-x_1           1               1
        """
        set1 = np.linspace(1.01, 5, 100)
        set3 = np.linspace(4.056, 5, 100)
        set4 = np.linspace(1.01, 3.732, 100)
        set5 = np.linspace(0.01, 1, 100)
        out =   [
                    {   "x": [5, 1, x, 0, 5, 0],
                        "f": [0] * self.num_of_objectives,
                        "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                    } for x in set1
                 ] + [
                        {   "x": [5, 1, x, 0, 1, 0],
                            "f": [0] * self.num_of_objectives,
                            "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                        } for x in set1
                ] + [
                        {   "x": [x, (x-2)/3, 1, 0, 1, 0],
                            "f": [0] * self.num_of_objectives,
                            "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                        } for x in set3
                ] + [
                        {   "x": [0, 2, x, 0, 1, 0],
                            "f": [0] * self.num_of_objectives,
                            "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                        } for x in set4
                ] + [
                        {   "x": [x, 2-x, 1, 0, 1, 0],
                            "f": [0] * self.num_of_objectives,
                            "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None
                        } for x in set5
                ]
        for o in out:
            self.evaluate(o["x"], o)
        return out