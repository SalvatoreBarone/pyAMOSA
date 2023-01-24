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


class TNK(pyamosa.Problem):
    def __init__(self):
        pyamosa.Problem.__init__(self, 2, [pyamosa.Type.REAL] * 2, [0.0000001] * 2, [np.pi] * 2, 2, 2)

    def evaluate(self, x, out):
        f1 = x[0]
        f2 = x[1]
        g1 = 1 + 0.1 * np.cos( 16 * np.arctan(x[0] / x[1])) - x[0] ** 2 - x[1] ** 2
        g2 = (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.5
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]

    def optimums(self):
        set1 = np.linspace(1e-6, np.pi, 100)
        set2 = np.linspace(1e-6, np.pi, 100)
        out = []
        for x1 in set1:
            for x2 in set2:
                if ((1 + 0.1 * np.cos(16 * np.arctan(x1 / x2)) - x1 ** 2 - x2 ** 2) <= 0) and (((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5) <= 0):
                    out.append({"x": [x1, x2], "f": [0] * self.num_of_objectives, "g": [0] * self.num_of_constraints if self.num_of_constraints > 0 else None})
        for o in out:
            self.evaluate(o["x"], o)
        return out


