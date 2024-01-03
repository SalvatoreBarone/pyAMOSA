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
import sys, os, copy, random, numpy as np
from tqdm import tqdm
from pyamosa.Config import Config
from .Optimizer import Optimizer
from .DataType import Type
from .Problem import Problem
from .StochasticHillClimbing import StochasticHillClimbing
from .StopCriterion import StopCriterion
from .VariableGrouping import VariableGrouping

"""
This optimizer class is intended to be used to tacke large scale oprimization problems.
For further insights, please refer to

    Song, An, Qiang Yang, Wei-Neng Chen, e Jun Zhang. "A random-based dynamic grouping strategy 
    for large scale multi-objective optimization". In 2016 IEEE Congress on Evolutionary 
    Computation (CEC), 468-75, 2016. https://doi.org/10.1109/CEC.2016.7743831.
    
"""
class GenericGroupingOptimizer(Optimizer):
    def __init__(self, config: Config, grouping_strategy : VariableGrouping):
        super().__init__(config)
        self.grouping_strategy = grouping_strategy
        self.pool_size = 0
        self.variable_masks = []
        self.current_mask_index = 0

    def initial_stage(self, problem, improve, remove_checkpoints):
        print("Initializing Variable Grouping")
        self.init_variable_grouping()

        climber = StochasticHillClimbing(problem, self.archive, self.config.hill_climb_checkpoint_file)
        if os.path.exists(self.config.minimize_checkpoint_file):
            print(f"Recovering Annealing from {self.config.minimize_checkpoint_file}")
            self.read_checkpoint(problem)
            problem.archive_to_cache(self.archive)
        elif improve is not None:
            print(f"Reading {improve}, and trying to improve a previous run...")
            self.archive.read_json(problem, improve)
            problem.archive_to_cache(self.archive)
            self.run_hill_climbing(climber, problem)
        elif os.path.exists(self.config.hill_climb_checkpoint_file):
            print(f"Recovering Hill-climbing from {self.config.hill_climb_checkpoint_file}")
            climber.read_checkpoint()
            print(f"Recovered {self.archive.size()} candidate solutions")
            self.run_hill_climbing(climber, problem)
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        else:
            climber.init()
            self.run_hill_climbing(climber, problem)
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        assert self.archive.size() > 0, "Archive not initialized"

    def annealing_loop(self, problem : Problem, termination_criterion : StopCriterion):
        assert self.archive.size() > 0, "Archive not initialized"
        tot_iterations = self.tot_iterations(termination_criterion)
        self.print_header(problem.num_of_constraints)
        self.print_statistics(problem.num_of_constraints)
        current_point = self.archive.random_point()
        pbar = tqdm(total = tot_iterations, desc = "Cooling the matter: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
        while not termination_criterion.check_termination(self):
            self.current_temperature *= self.config.cooling_factor
            self.select_group()
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

    def init_variable_grouping(self):
        if len(self.grouping_strategy.variable_masks) == 0:
            self.grouping_strategy.run()
        self.variable_masks = self.grouping_strategy.variable_masks
        self.pool_size = len(self.variable_masks)
        self.groups_score = np.ones(self.pool_size)
        
    def select_group(self):
        self.current_group_index = random.choices(list(range(self.pool_size)), weights = Optimizer.softmax(7 * self.groups_score), k=1)[0]

    def update_group_score(self):
        self.groups_score[self.current_group_index] = self.archive.C_actual_prev

    def random_perturbation(self, problem, s, strength):
        z = copy.deepcopy(s)
        safety_exit = problem.max_attempt
        while safety_exit >= 0 and problem.is_cached(z):
            safety_exit -= 1
            selected_indexes = random.choices(list(range(problem.num_of_variables)), weights = self.variable_masks[self.current_mask_index], k = random.randrange(1, 1 + min([strength, problem.num_of_variables])))
            allowed_indexes = np.nonzero(self.variable_masks[self.current_mask_index])[0].tolist()
            assert all(i in allowed_indexes for i in selected_indexes), f"One of the selected decision variables must not be altered.\nSelected indexes {selected_indexes}\nAllowed variables: {allowed_indexes}\n"
            for i in selected_indexes:
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