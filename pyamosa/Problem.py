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
import numpy as np
from .MultiFileCacheHandle import MultiFileCacheHandle
from .DataType import Type

class Problem:
    def __init__(self, num_of_variables : int , types : list, lower_bounds : list, upper_bounds : list, num_of_objectives : int, num_of_constraints : int):
        assert num_of_variables == len(types), "Mismatch in the specified number of variables and their type declaration"
        assert num_of_variables == len(lower_bounds), "Mismatch in the specified number of variables and their lower bound declaration"
        assert num_of_variables == len(upper_bounds), "Mismatch in the specified number of variables and their upper bound declaration"
        self.num_of_variables = num_of_variables
        self.num_of_objectives = num_of_objectives
        self.num_of_constraints = num_of_constraints
        for t in types:
            assert t in [Type.INTEGER, Type.REAL], "Only AMOSA.Type.INTEGER or AMOSA.Type.REAL data-types for decison variables are supported!"
        self.types = types
        self.min_step = [1 if t == Type.INTEGER else (2 * np.finfo(float).eps) for t in self.types]
        for lb, ub, t in zip(lower_bounds, upper_bounds, self.types):
            assert isinstance(lb, int if t == Type.INTEGER else float), f"Type mismatch. Value {lb} in lower_bound is not suitable for {t}"
            assert isinstance(ub, int if t == Type.INTEGER else float), f"Type mismatch. Value {ub} in upper_bound is not suitable for {t}"
        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds
        self.cache = {}
        self.total_calls = 0
        self.cache_hits = 0
        self.max_attempt = self.num_of_variables

    def evaluate(self, x : list, out : dict):
        pass

    def optimums(self):
        return []

    @staticmethod
    def get_cache_key(s):
        return ''.join([str(i) for i in s["x"]])

    def is_cached(self, s):
        return self.get_cache_key(s) in self.cache.keys()

    def add_to_cache(self, s):
        self.cache[self.get_cache_key(s)] = {"f": s["f"], "g": s["g"]}

    def load_cache(self, directory):
        handler = MultiFileCacheHandle(directory)
        self.cache = handler.read()

    def store_cache(self, directory):
        handler = MultiFileCacheHandle(directory)
        handler.write(self.cache)

    def archive_to_cache(self, archive):
        for s in archive:
            if not self.is_cached(s):
                self.add_to_cache(s)