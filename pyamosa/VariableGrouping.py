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
import numpy as np, json5
from tqdm import trange
from .DataType import Type
from .Problem import Problem

class VariableGrouping:
    def __init__(self, problem : Problem, cache : str = "grouping_cache.json5"):
        self.problem = problem
        self.cache_file = cache
        self.variable_groups = []

    def load(self):
        with open(self.cache_file) as file:
            self.variable_groups = json5.load(file)

    def store(self):
        try:
            with open(self.cache_file, 'w') as outfile:
                json5.dump(self.variable_groups, outfile)
        except TypeError as e:
            print(self.variable_groups)
            print(e)
            exit()

    def min_step(self, dimension):
        return 1 if self.problem.types[dimension] == Type.INTEGER else (5 * np.finfo(float).eps)

    def run(self, problem_cache : str, eps : float = 10 * np.finfo(float).eps):
        pass