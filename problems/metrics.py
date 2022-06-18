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
import numpy as np


def coverage_sets(set_A, set_B):
    count = 0
    for b in set_B:
        for a in set_A:
            if all(a <= b) and any (a < b):
                count = count + 1
                break
    return count / len(set_B)


def convergence(R_star, R):
    return sum([ min([ np.linalg.norm(r - r_star) for r_star in R_star]) for r in R ]) / len(R)


def dispersion(R_star, R):
    return sum([min([np.linalg.norm(r - r_star) for r in R]) for r_star in R_star]) / len(R_star)