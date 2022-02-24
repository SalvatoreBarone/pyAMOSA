#!/usr/bin/python3
"""
Copyright 2021 Salvatore Barone <salvatore.barone@unina.it>

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
from AMOSA import *

class OSY(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 6, [AMOSA.Type.REAL] * 6, [0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10], 2, 6)

    def evaluate(self, x, out):
        f1 = -(25 * (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 + (x[3] - 4) ** 2 + (x[4] - 1) ** 2 )
        f2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2
        g1 = 2 - x[0] - x[1]
        g2 = x[0] + x[1] -6
        g3 = x[1] - x[0] - 2
        g4 = x[0] - 3 * x[1] - 2
        g5 = x[3]  + (x[2] - 3) ** 2 - 4
        g6 = 4 - x[5] - (x[4] - 3) ** 2
        out["f"] = [f1, f2]
        out["g"] = [g1, g2, g2, g3, g4, g5, g6]

if __name__ == '__main__':
    config = AMOSAConfig
    config.archive_hard_limit = 75
    config.archive_soft_limit = 150
    config.archive_gamma = 2
    config.hill_climbing_iterations = 2500
    config.initial_temperature = 500
    config.final_temperature = 0.0000001
    config.cooling_factor = 0.9
    config.annealing_iterations = 2500
    config.early_terminator_window = 15

    problem = OSY()
    optimizer = AMOSA(config)
    optimizer.minimize(problem)
    optimizer.save_results(problem, "osy.csv")
    optimizer.plot_pareto(problem, "osy.pdf")

