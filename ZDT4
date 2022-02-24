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

class ZDT4(AMOSA.Problem):
    def __init__(self):
        AMOSA.Problem.__init__(self, 10, [AMOSA.Type.REAL] * 10, [0, -10, -10, -10, -10, -10, -10, -10, -10, -10], [1, 10, 10, 10, 10, 10, 10, 10, 10, 10], 2, 0)

    def evaluate(self, x, out):
        f = x[0]
        g = 1 + 10 * 9 + sum( [ i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:] ] )
        h = 1 - np.sqrt(f / g)
        out["f"] = [f, g * h ]
        pass


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

    problem = ZDT4()
    optimizer = AMOSA(config)
    optimizer.minimize(problem)
    optimizer.save_results(problem, "zdt4.csv")
    optimizer.plot_pareto(problem, "zdt4.pdf")

