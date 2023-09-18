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
import copy, numpy as np
from tqdm import trange
from .Problem import Problem
from .VariableGrouping import VariableGrouping

class DifferentialVariableGrouping(VariableGrouping):
    def __init__(self, problem : Problem, cache : str = "differential_grouping_cache.json5"):
        super().__init__(problem, cache)

    def run(self, problem_cache : str, eps : float = 10 * np.finfo(float).eps):
        lower_point = self.problem.lower_point()
        interacting_variables = {}
        separable_variables = []
        for i in trange(self.problem.num_of_variables,  desc = "Checking epistasis: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            if all(i not in g for g in interacting_variables.values()): # check whether i has been already grouped
                x = copy.deepcopy(lower_point)
                x["x"][i] += self.min_step(i)
                delta_lb = np.array(x["f"]) - np.array(lower_point["f"])
                self.problem.get_objectives(x)
                for j in range(self.problem.num_of_variables):
                    if i != j and j not in interacting_variables.keys() and all(j not in g for g in interacting_variables.values()) and j not in separable_variables: # check whether j differs from i and j has not been already grouped
                        y = copy.deepcopy(lower_point)
                        y["x"][j] = self.problem.upper_bound[j] - self.min_step(j)
                        self.problem.get_objectives(y)
                        z = copy.deepcopy(x)
                        z["x"][j] = y["x"][j]
                        self.problem.get_objectives(z)
                        delta_ub = np.array(z["f"]) - np.array(y["f"])
                        if any(np.abs(delta_lb - delta_ub) > eps):
                            if i in interacting_variables:
                                interacting_variables[i].append(j)
                            else:
                                interacting_variables[i] = [j]
                if i not in interacting_variables: # if i does not interact, it is a separable variable
                    separable_variables.append(i)
        self.problem.store_cache(problem_cache)
        interacting_variables = [ [k] + v for k, v in interacting_variables.items() ] 
        self.variable_groups = interacting_variables + [separable_variables] if separable_variables else interacting_variables
        



