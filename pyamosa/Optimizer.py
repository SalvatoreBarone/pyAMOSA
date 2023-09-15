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
import sys, copy, random, time, os, json5, warnings, numpy as np
from tqdm import tqdm, trange
from .DataType import Type
from .Config import Config
from .Problem import Problem
from .Pareto import Pareto
from .HillClimbing import HillClimbing
from .StopCriterion import StopCriterion
from .StopMinTemperature import StopMinTemperature
from .CombinedStopCriterion import CombinedStopCriterion
class Optimizer:
    checkpoint_file = "minimize_checkpoint.json5"
    

    def __init__(self, config : Config):
        warnings.filterwarnings("error")
        self.config = config

        self.current_temperature = 0
        self.archive = Pareto()
        self.duration = 0
        self.n_eval = 0


    def run(self, problem : Problem, termination_criterion : StopCriterion = StopMinTemperature(1e-10), improve : str = None, remove_checkpoints : bool = True):
        print(f"Reading cache from {self.config.cache_dir}. This may take a while...")
        problem.load_cache(self.config.cache_dir)
        print(f"Read {len(problem.cache)} entries.")
        self.current_temperature = self.config.initial_temperature
        # self.temperature = Optimizer.matter_temperatures(self.config.initial_temperature, self.final_temperature, self.config.cooling_factor)

        assert self.config.annealing_strength <= problem.num_of_variables, f"Too much strength ({self.config.annealing_strength}) for this problem! It has only {problem.num_of_variables} variables!"
        self.archive = Pareto()
        self.duration = 0
        
        self.duration = time.time()
        if os.path.exists(self.config.minimize_checkpoint_file):
            print(f"Recovering Annealing from {self.config.minimize_checkpoint_file}")
            self.read_checkpoint(problem)
            problem.archive_to_cache(self.archive)
        elif improve is not None:
            self.archive.read_json(problem.types, improve)
            problem.archive_to_cache(self.archive)
            if self.archive.size() > self.config.archive_hard_limit:
                self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
            self.save_checkpoint()
            
            
        elif os.path.exists(self.config.hill_climb_checkpoint_file):
            print(f"Recovering Hill-climbing from {self.config.hill_climb_checkpoint_file}")
            climber = HillClimbing(problem, self.archive, self.config.hill_climb_checkpoint_file)
            climber.read_checkpoint()
            print(f"Recovered {self.archive.size()} candidate solutions")
            climber.run(self.config.archive_soft_limit * self.config.archive_gamma, self.config.annealing_iterations)
            problem.archive_to_cache(self.archive)
            if self.archive.size() > self.config.archive_hard_limit:
                self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
            self.save_checkpoint()
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
                
        else:
            climber = HillClimbing(problem, self.archive, self.config.hill_climb_checkpoint_file)
            climber.init()
            climber.run(self.config.archive_soft_limit * self.config.archive_gamma, self.config.annealing_iterations)
            problem.archive_to_cache(self.archive)
            if self.archive.size() > self.config.archive_hard_limit:
                self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
            self.save_checkpoint()
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        assert self.archive.size() > 0, "Archive not initialized"
        
        self.main_loop(problem, termination_criterion)
        self.archive.remove_infeasible(problem.num_of_constraints)
        self.archive.remove_dominated()
        if self.archive.size() > self.config.archive_hard_limit:
            self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
        self.print_statistics(problem)
        self.duration = time.time() - self.duration
        problem.store_cache(self.config.cache_dir)
        if remove_checkpoints:
            os.remove(self.config.minimize_checkpoint_file)

    def main_loop(self, problem : Problem, termination_criterion : StopCriterion):
        tot_iterations = self.tot_iterations(termination_criterion)
        self.print_header(problem)
        self.print_statistics(problem)
        current_point = self.archive.random_point()
        pbar = tqdm(total = tot_iterations, desc = "Main loop: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
        while not termination_criterion.check_termination(self):
            self.current_temperature *= self.config.cooling_factor
            self.archive = Optimizer.annealing_thread_loop(problem, self.archive, current_point, self.current_temperature, self.config.annealing_iterations, self.config.annealing_strength, self.config.archive_soft_limit, self.config.archive_hard_limit, self.config.clustering_max_iterations, False, True)
            self.n_eval += self.config.annealing_iterations
            self.print_statistics(problem)
            if self.archive.size() > self.config.archive_soft_limit:
                self.archive.clustering(problem.num_of_constraints, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
                self.print_statistics(problem)
            self.save_checkpoint()
            problem.store_cache(self.config.cache_dir)
            pbar.update(1)
        print("Termination criterion has been met.")

    def tot_iterations(self, termination_criterion : StopCriterion):
        min_temperature = 1e-10
        if isinstance(termination_criterion, StopMinTemperature):
            min_temperature = termination_criterion.min_temperature
        elif isinstance(termination_criterion, CombinedStopCriterion):
            min_temperature = termination_criterion.min_temperat.min_temperature
        return int(np.ceil(np.log(min_temperature/self.current_temperature) / np.log(self.config.cooling_factor)))

    @staticmethod
    def annealing_thread_loop(problem, archive, current_point, current_temperature, annealing_iterations, annealing_strength, soft_limit, hard_limit, clustering_max_iterations, clustering_before_return, print_allowed):
        for _ in trange(annealing_iterations, desc = "Annealing", file=sys.stdout, leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}") if print_allowed else range(annealing_iterations):
            new_point = Optimizer.random_perturbation(problem, current_point, annealing_strength)
            fitness_range = archive.compute_fitness_range([current_point, new_point])
            s_dominating_y = archive.dominating(new_point)
            s_dominated_by_y = archive.dominated_by(new_point)
            k_s_dominated_by_y = len(s_dominated_by_y)
            k_s_dominating_y = len(s_dominating_y)
            if Pareto.dominates(current_point, new_point) and k_s_dominating_y >= 0:
                delta_avg = (np.nansum([Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]) + Optimizer.domination_amount(current_point, new_point, fitness_range)) / (k_s_dominating_y + 1)
                if Optimizer.accept(Optimizer.sigmoid(-delta_avg * current_temperature)):
                    current_point = new_point
            elif not Pareto.dominates(current_point, new_point) and not Pareto.dominates(new_point, current_point):
                if k_s_dominating_y >= 1:
                    delta_avg = np.nansum([Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]) / k_s_dominating_y
                    if Optimizer.accept(Optimizer.sigmoid(-delta_avg * current_temperature)):
                        current_point = new_point
                elif (k_s_dominating_y == 0 and k_s_dominated_by_y == 0) or k_s_dominated_by_y >= 1:
                    archive.add(new_point)
                    current_point = new_point
                    if archive.size() > soft_limit:
                        archive.clustering(problem.num_of_constraints, hard_limit, clustering_max_iterations, print_allowed)
            elif Pareto.dominates(new_point, current_point):
                if k_s_dominating_y >= 1:
                    delta_dom = [Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]
                    if Optimizer.accept(Optimizer.sigmoid(min(delta_dom))):
                        current_point = archive[np.argmin(delta_dom)]
                elif (k_s_dominating_y == 0 and k_s_dominated_by_y == 0) or k_s_dominated_by_y >= 1:
                    archive.add(new_point)
                    current_point = new_point
                    if archive.size() > soft_limit:
                        archive.clustering(problem.num_of_constraints, hard_limit, clustering_max_iterations, print_allowed)
            else:
                raise RuntimeError(f"Something went wrong\narchive: {archive}\nx:{current_point}\ny: {new_point}\n x < y: {Pareto.dominates(current_point, new_point)}\n y < x: {Pareto.dominates(new_point, current_point)}\ny domination rank: {k_s_dominated_by_y}\narchive domination rank: {k_s_dominating_y}")
        archive.clustering(problem.num_of_constraints, hard_limit, clustering_max_iterations, print_allowed) if clustering_before_return else archive
        return archive

    def print_header(self, problem):
        if problem.num_of_constraints == 0:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>10} | {:>10} | {:>10} |".format("temp.", "# eval", " # nds", "D*", "Dnad", "phi"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10))
        else:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |".format("temp.", "# eval", "# nds", "# feas", "cv min", "cv avg", "D*", "Dnad", "phi"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))

    def print_statistics(self, num_of_constraints):
        delta_nad, delta_ideal, phy = self.archive.compute_deltas()
        if num_of_constraints == 0:
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>10.3e} | {:>10.3e} | {:>10.3e} |".format(self.current_temperature, self.n_eval, self.archive.size(), delta_ideal, delta_nad, phy))
        else:
            feasible, cv_min, cv_avg = self.archive.get_min_agv_cv()
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>6} | {:>10.2e} | {:>10.2e} | {:>10.3e} | {:>10.3e} | {:>10.3e} |".format(self.current_temperature, self.n_eval, self.archive.size(), feasible, cv_min, cv_avg, delta_ideal, delta_nad, phy))

    def save_checkpoint(self):
        checkpoint = {"n_eval": self.n_eval, "t": self.current_temperature} | self.archive.get_checkpoint()
        try:
            with open(self.config.minimize_checkpoint_file, 'w') as outfile:
                json5.dump(checkpoint, outfile)
        except TypeError as e:
            print(checkpoint)
            print(e)
            exit()

    def read_checkpoint(self, problem):
        with open(self.config.minimize_checkpoint_file) as file:
            checkpoint = json5.load(file)
        self.n_eval = int(checkpoint["n_eval"])
        self.current_temperature = float(checkpoint["t"])
        self.archive.from_checkpoint(checkpoint, problem.types)
        
    @staticmethod
    def accept(probability):
        return random.random() < probability

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(np.array(-x, dtype = np.float128)))
    
    @staticmethod
    def domination_amount(x, y, r):
        return np.prod([abs(i - j) / k if k != 0 else 0 for i, j, k in zip(x["f"], y["f"], r)])

    @staticmethod
    def random_perturbation(problem, s, strength):
        z = copy.deepcopy(s)
        # while z["x"] is in the cache, repeat the random perturbation a safety-exit prevents infinite loop, using a counter variable
        safety_exit = problem.max_attempt
        while safety_exit >= 0 and problem.is_cached(z):
            safety_exit -= 1
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


