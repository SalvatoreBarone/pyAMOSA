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
        self.pool_size = 0
        self.group_size_pool = []
        self.group_size_score = []
        self.current_group_index = 0
        self.current_variable_mask = []

    def annealing_loop(self, problem : Problem, termination_criterion : StopCriterion):
        assert self.archive.size() > 0, "Archive not initialized"
        tot_iterations = self.tot_iterations(termination_criterion)
        print("Initializing Dynamic Random Grouping")
        self.init_variable_grouping(problem, tot_iterations)
        self.print_header(problem.num_of_constraints)
        self.print_statistics(problem.num_of_constraints)
        current_point = self.archive.random_point()
        pbar = tqdm(total = tot_iterations, desc = "Cooling the matter: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
        while not termination_criterion.check_termination(self):
            self.current_temperature *= self.config.cooling_factor
            self.random_variable_grouping(problem.num_of_variables)
            self.annealing(problem, current_point)
            self.update_group_score()
            self.n_eval += self.config.annealing_iterations
            self.print_statistics(problem.num_of_constraints)
            if self.archive.size() > self.config.archive_soft_limit:
                self.archive.clustering(problem, self.config.archive_hard_limit, self.config.clustering_max_iterations)
                self.print_statistics(problem.num_of_constraints)
            self.save_checkpoint()
            problem.store_cache(self.config.cache_dir)
            pbar.update(1)
        print("Termination criterion has been met.")

    @staticmethod
    def softmax(x):
        e_x = np.exp(np.array(x, dtype = np.float64))
        return e_x / e_x.sum()
    
    def init_variable_grouping(self, problem, tot_iterations):
        self.pool_size = min(int(0.6 * tot_iterations), 15)
        self.group_size_pool = np.geomspace(5*int(np.log10(problem.num_of_variables)), problem.num_of_variables // 2, num = self.pool_size, endpoint = True, dtype = int).tolist()
        self.group_size_score = np.ones(self.pool_size)
        self.current_group_index = self.pool_size - 1
        self.current_variable_mask = [1] * self.pool_size
        print(f"Pool: {self.group_size_pool} (size: {self.pool_size})")

    def random_variable_grouping(self, num_of_variables):
        self.current_group_index = random.choices(list(range(self.pool_size)), weights = Optimizer.softmax(7 * self.group_size_score), k=1)[0]
        self.current_variable_mask = [0] * (num_of_variables - self.group_size_pool[self.current_group_index]) + [1] * self.group_size_pool[self.current_group_index]
        random.shuffle(self.current_variable_mask)
        #print(f"Current index: {self.current_group_index}, current size: {self.group_size_pool[self.current_group_index]}, current mask: {self.current_variable_mask}")
        assert np.sum(self.current_variable_mask) == self.group_size_pool[self.current_group_index], f"{np.sum(self.current_variable_mask)} differs from {self.group_size_pool[self.current_group_index]}"

    def update_group_score(self):
        self.group_size_score[self.current_group_index] = self.archive.C_actual_prev

    def random_perturbation(self, problem, s, strength):
        z = copy.deepcopy(s)
        safety_exit = problem.max_attempt
        while safety_exit >= 0 and problem.is_cached(z):
            safety_exit -= 1
            indexes = random.choices(list(range(problem.num_of_variables)), weights = self.current_variable_mask, k = random.randrange(1, 1 + min([strength, problem.num_of_variables])))
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
    
    def print_header(self, num_of_constraints):
        if num_of_constraints == 0:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>6}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 6))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>6} |".format("temp.", "# eval", " # nds", "D*", "Dnad", "phi", "C(P', P)", "C(P, P')", "S"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>6}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 6))
        else:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>6}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 6))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>6} |".format("temp.", "# eval", "# nds", "# feas", "cv min", "cv avg", "D*", "Dnad", "phi", "C(P', P)", "C(P, P')", "S"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>6}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 6))

    def print_statistics(self, num_of_constraints):
        delta_nad, delta_ideal, phy, C_prev_actual, C_actual_prev = self.archive.compute_deltas()
        if num_of_constraints == 0:
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>6} |".format(self.current_temperature, self.n_eval, self.archive.size(), delta_ideal, delta_nad, phy, C_prev_actual, C_actual_prev, self.group_size_pool[self.current_group_index]))
        else:
            feasible, cv_min, cv_avg = self.archive.get_min_agv_cv()
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>6} | {:>10.2e} | {:>10.2e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>10.3e} | {:>6} |".format(self.current_temperature, self.n_eval, self.archive.size(), feasible, cv_min, cv_avg, delta_ideal, delta_nad, phy, C_prev_actual, C_actual_prev, self.group_size_pool[self.current_group_index]))