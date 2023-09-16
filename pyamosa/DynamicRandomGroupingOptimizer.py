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
import copy, random, numpy as np
from tqdm import tqdm
from pyamosa.Config import Config
from .Optimizer import Optimizer
from .DataType import Type
from .Problem import Problem
from .Pareto import Pareto
from .StopCriterion import StopCriterion

"""
This optimizer class is intended to be used to tacke large scale oprimization problems.
For further insights, please refer to

    Song, An, Qiang Yang, Wei-Neng Chen, e Jun Zhang. "A random-based dynamic grouping strategy 
    for large scale multi-objective optimization". In 2016 IEEE Congress on Evolutionary 
    Computation (CEC), 468-75, 2016. https://doi.org/10.1109/CEC.2016.7743831.
    
"""
class DynamicRandomGroupingOptimizer(Optimizer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.group_size_pool = []
        self.group_size_score = []
        self.group_size = 0
        self.variable_subset = []

    def main_loop(self, problem : Problem, termination_criterion : StopCriterion):
        assert self.archive.size() > 0, "Archive not initialized"
        tot_iterations = self.tot_iterations(termination_criterion)
        self.print_header(problem)
        self.print_statistics(problem)
        current_point = self.archive.random_point()
        pbar = tqdm(total = tot_iterations, desc = "Cooling the matter...", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
        while not termination_criterion.check_termination(self):
            self.current_temperature *= self.config.cooling_factor
            # TODO define which subset of variables has to be altered and its size
            self.annealing_loop(problem, current_point)
            self.n_eval += self.config.annealing_iterations
            self.print_statistics(problem)
            if self.archive.size() > self.config.archive_soft_limit:
                self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations)
                self.print_statistics(problem)
            self.save_checkpoint()
            problem.store_cache(self.config.cache_dir)
            pbar.update(1)
        print("Termination criterion has been met.")

    def random_variable_grouping(self):
        # TODO define which subset of variables has to be altered and its size
        pass

    def random_perturbation(self, problem, s, strength):
        z = copy.deepcopy(s)
        # while z["x"] is in the cache, repeat the random perturbation a safety-exit prevents infinite loop, using a counter variable
        safety_exit = problem.max_attempt
        while safety_exit >= 0 and problem.is_cached(z):
            safety_exit -= 1
            # TODO select indexes only from the list of allowed indexes
            indexes = random.sample(range(problem.num_of_variables), random.randrange(1, 1 + min([strength, problem.num_of_variables])))
            for i in indexes:
                lb = problem.lower_bound[i]
                ub = problem.upper_bound[i]
                tp = problem.types[i]
                narrow_interval = ((ub - lb) == 1) if tp == Type.INTEGER else ((ub - lb) <= np.finfo(float).eps)
                if narrow_interval:
                    z["x"][i] = lb
                else:
                    z["x"][i] = random.randrange(lb, ub) if tp == Type.INTEGER else random.uniform(lb, ub)
        problem.get_objectives(z)
        return z