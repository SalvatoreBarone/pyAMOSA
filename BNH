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

class BNH(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 2, [AMOSA.Type.REAL] * 2, [0]*2, [5, 3], 2, 2)

    def evaluate(self, x, out):
        f1 = 4 * x[0] ** 2 + 4 * x[1] ** 2
        f2 = (x[0] - 5) ** 2 + (x[1] - 5) ** 2
        g1 = (x[0] - 5) ** 2 + x[1] ** 2 - 25
        g2 = 7.7 - (x[0] - 5) ** 2 - (x[1] + 3) ** 2
        out["f"] = [f1, f2 ]
        out["g"] = [g1, g2]

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

    problem = BNH()
    optimizer = AMOSA(config)
    optimizer.minimize(problem)
    optimizer.save_results(problem, "bnh.csv")
    optimizer.plot_pareto(problem, "bnh.pdf")

