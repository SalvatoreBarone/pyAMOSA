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
import sys, copy, random, time, os, json, warnings, numpy as np, matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from tqdm import tqdm, trange
from .DataType import Type
from .Config import Config
from .Problem import Problem
from .StopCriterion import StopCriterion
from .StopMinTemperature import StopMinTemperature
class Optimizer:
    hill_climb_checkpoint_file = "hill_climb_checkpoint.json"
    minimize_checkpoint_file = "minimize_checkpoint.json"
    cache_dir = ".cache"

    def __init__(self, config : Config):
        warnings.filterwarnings("error")
        self.config = config

        self.current_temperature = 0
        self.archive = []
        self.duration = 0
        self.n_eval = 0
        self.ideal = None
        self.nadir = None
        self.old_norm_objectives = []
        self.phy = []

    def run(self, problem : Problem, termination_criterion : StopCriterion = StopMinTemperature(1e-10), improve : str = None, remove_checkpoints : bool = True):
        problem.load_cache(self.config.cache_dir)
        self.current_temperature = self.config.initial_temperature
        # self.temperature = Optimizer.matter_temperatures(self.config.initial_temperature, self.final_temperature, self.config.cooling_factor)

        assert self.config.annealing_strength <= problem.num_of_variables, f"Too much strength ({self.config.annealing_strength}) for this problem! It has only {problem.num_of_variables} variables!"
        self.archive = []
        self.duration = 0
        self.n_eval = 0
        self.ideal = None
        self.nadir = None
        self.old_norm_objectives = []
        self.phy = []

        self.duration = time.time()
        if os.path.exists(self.config.minimize_checkpoint_file):
            self.read_checkpoint_minimize(problem)
            problem.archive_to_cache(self.archive)
        elif os.path.exists(self.config.hill_climb_checkpoint_file):
            initial_candidate = self.read_checkpoint_hill_climb(problem)
            problem.archive_to_cache(initial_candidate)
            self.initial_hill_climbing(problem, initial_candidate)
            if len(self.archive) > self.config.archive_hard_limit:
                self.archive = Optimizer.clustering(self.archive, problem, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
            self.save_checkpoint_minimize()
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        elif improve is not None:
            self.archive_from_json(problem, improve)
            problem.archive_to_cache(self.archive)
            if len(self.archive) > self.config.archive_hard_limit:
                self.archive = Optimizer.clustering(self.archive, problem, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
            self.save_checkpoint_minimize()
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        else:
            self.random_archive(problem)
            self.save_checkpoint_minimize()
            if remove_checkpoints:
                os.remove(self.config.hill_climb_checkpoint_file)
        assert len(self.archive) > 0, "Archive not initialized"
        Optimizer.print_header(problem)
        self.print_statistics(problem)
        self.main_loop(problem, termination_criterion)
        self.archive = Optimizer.remove_infeasible(problem, self.archive)
        self.archive = Optimizer.remove_dominated(self.archive)
        if len(self.archive) > self.config.archive_hard_limit:
            self.archive = Optimizer.clustering(self.archive, problem, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
        self.print_statistics(problem)
        self.duration = time.time() - self.duration
        problem.store_cache(self.config.cache_dir)
        if remove_checkpoints:
            os.remove(self.config.minimize_checkpoint_file)

    def pareto_front(self):
        return np.array([s["f"] for s in self.archive])

    def pareto_set(self):
        return np.array([s["x"] for s in self.archive])

    def constraint_violation(self):
        return np.array([s["g"] for s in self.archive])

    def plot_pareto(self, problem : Problem, pdf_file : str, fig_title : str = "Pareto front", axis_labels : list = None):
        if axis_labels is None:
            axis_labels = [f"f{str(i)}" for i in range(problem.num_of_objectives)]
        F = self.pareto_front()
        if problem.num_of_objectives == 2:
            plt.figure(figsize = (10, 10), dpi = 300)
            plt.plot(F[:, 0], F[:, 1], 'k.')
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            plt.title(fig_title)
            plt.savefig(pdf_file, bbox_inches = 'tight', pad_inches = 0)
        elif problem.num_of_objectives == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            plt.title(fig_title)
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], marker = '.', color = 'k')
            plt.tight_layout()
            plt.savefig(pdf_file, bbox_inches = 'tight', pad_inches = 0)

    def archive_to_json(self, json_file : str):
        try:
            with open(json_file, 'w') as outfile:
                outfile.write(json.dumps(self.archive))
        except TypeError as e:
            print(self.archive)
            print(e)
            exit()

    def archive_to_csv(self, problem : Problem, csv_file : str, fitness_labels : list = None):
        original_stdout = sys.stdout
        row_format = "{:};" + "{:};" * problem.num_of_objectives + "{:};" * problem.num_of_variables
        if fitness_labels is None:
            fitness_labels = [f"f{i}" for i in range(problem.num_of_objectives)]	
        with open(csv_file, "w") as file:
            sys.stdout = file
            print(row_format.format("", *fitness_labels, *[f"x{i}" for i in range(problem.num_of_variables)]))
            for i, f, x in zip(range(len(self.pareto_front())), self.pareto_front(), self.pareto_set()):
                print(row_format.format(i, *f, *x))
        sys.stdout = original_stdout

    def random_archive(self, problem):
        print("Initializing random archive...")
        #initial_candidate_solutions = [Optimizer.lower_point(problem), Optimizer.upper_point(problem)]
        initial_candidate_solutions = [Optimizer.lower_point(problem)]
        self.initial_hill_climbing(problem, initial_candidate_solutions)

    def archive_from_json(self, problem, json_file):
        print("Initializing archive from JSON file...")
        with open(json_file) as f:
            archive = json.load(f)
        initial_candidate_solutions = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in archive]
        self.initial_hill_climbing(problem, initial_candidate_solutions)

    def read_final_archive_from_json(self, problem : Problem, json_file : str):
        print("Reading archive from JSON file...")
        with open(json_file) as file:
            archive = json.load(file)
        self.archive = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in archive]

    def initial_hill_climbing(self, problem, initial_candidate_solutions):
        num_of_initial_candidate_solutions = self.config.archive_gamma * self.config.archive_soft_limit
        if self.config.hill_climbing_iterations > 0:
            if self.config.multiprocessing_enabled:
                args = [[problem, self.config.hill_climbing_iterations]] * cpu_count()
                print(f"Performing Initial Hill Climbing Step using {cpu_count()} threads")
                for _ in trange(len(initial_candidate_solutions), num_of_initial_candidate_solutions, cpu_count(), desc = "Hill climbing", leave = False):
                    with Pool(cpu_count()) as pool:
                        new_points = pool.starmap(Optimizer.hillclimb_thread_loop, args)
                    initial_candidate_solutions += new_points
                    self.save_checkpoint_hillclimb(initial_candidate_solutions)
            else:
                for _ in trange(len(initial_candidate_solutions), num_of_initial_candidate_solutions, desc = "Hill climbing", leave = False):
                    initial_candidate_solutions.append(Optimizer.hillclimb_thread_loop(problem, self.config.hill_climbing_iterations))
                    self.save_checkpoint_hillclimb(initial_candidate_solutions)
        for x in initial_candidate_solutions:
            Optimizer.add_to_archive(self.archive, x)

    @staticmethod
    def hillclimb_thread_loop(problem, hillclimb_iterations):
        return Optimizer.hill_climbing(problem, Optimizer.random_point(problem), hillclimb_iterations)

    def main_loop(self, problem : Problem, termination_criterion : StopCriterion):
        current_point = random.choice(self.archive)
        while not termination_criterion.check_termination(self):
            self.current_temperature *= self.config.cooling_factor
            if self.config.multiprocessing_enabled:
                args = [[problem, self.archive.copy(), random.choice(self.archive), self.current_temperature, self.config.annealing_iterations, self.config.annealing_strength, self.config.archive_soft_limit, self.config.archive_hard_limit, self.config.clustering_max_iterations, True, i] for i in [t == 0 for t in range(cpu_count())]]
                with Pool(cpu_count()) as pool:
                    archives = pool.starmap(Optimizer.annealing_thread_loop, args)
                self.archive = Optimizer.nondominated_merge(archives)
                self.n_eval += self.config.annealing_iterations * cpu_count()
            else:
                self.archive = Optimizer.annealing_thread_loop(problem, self.archive, current_point, self.current_temperature, self.config.annealing_iterations, self.config.annealing_strength, self.config.archive_soft_limit, self.config.archive_hard_limit, self.config.clustering_max_iterations, False, True)
                self.n_eval += self.config.annealing_iterations
            self.print_statistics(problem)
            if len(self.archive) > self.config.archive_soft_limit:
                self.archive = Optimizer.clustering(self.archive, problem, self.config.archive_hard_limit, self.config.clustering_max_iterations, True)
                self.print_statistics(problem)
            self.save_checkpoint_minimize()
            problem.store_cache(self.config.cache_dir)
        print("Termination criterion has been met.")

    @staticmethod
    def annealing_thread_loop(problem, archive, current_point, current_temperature, annealing_iterations, annealing_strength, soft_limit, hard_limit, clustering_max_iterations, clustering_before_return, print_allowed):
        for _ in trange(annealing_iterations, desc = "Annealing", file=sys.stdout, leave = False) if print_allowed else range(annealing_iterations):
            new_point = Optimizer.random_perturbation(problem, current_point, annealing_strength)
            fitness_range = Optimizer.compute_fitness_range(archive, current_point, new_point)
            s_dominating_y = [s for s in archive if Optimizer.dominates(s, new_point)]
            s_dominated_by_y = [s for s in archive if Optimizer.dominates(new_point, s)]
            k_s_dominated_by_y = len(s_dominated_by_y)
            k_s_dominating_y = len(s_dominating_y)
            if Optimizer.dominates(current_point, new_point) and k_s_dominating_y >= 0:
                delta_avg = (np.nansum([Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]) + Optimizer.domination_amount(current_point, new_point, fitness_range)) / (k_s_dominating_y + 1)
                if Optimizer.accept(Optimizer.sigmoid(-delta_avg * current_temperature)):
                    current_point = new_point
            elif not Optimizer.dominates(current_point, new_point) and not Optimizer.dominates(new_point, current_point):
                if k_s_dominating_y >= 1:
                    delta_avg = np.nansum([Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]) / k_s_dominating_y
                    if Optimizer.accept(Optimizer.sigmoid(-delta_avg * current_temperature)):
                        current_point = new_point
                elif (k_s_dominating_y == 0 and k_s_dominated_by_y == 0) or k_s_dominated_by_y >= 1:
                    Optimizer.add_to_archive(archive, new_point)
                    current_point = new_point
                    if len(archive) > soft_limit:
                        archive = Optimizer.clustering(archive, problem, hard_limit, clustering_max_iterations, print_allowed)
            elif Optimizer.dominates(new_point, current_point):
                if k_s_dominating_y >= 1:
                    delta_dom = [Optimizer.domination_amount(s, new_point, fitness_range) for s in s_dominating_y]
                    if Optimizer.accept(Optimizer.sigmoid(min(delta_dom))):
                        current_point = archive[np.argmin(delta_dom)]
                elif (k_s_dominating_y == 0 and k_s_dominated_by_y == 0) or k_s_dominated_by_y >= 1:
                    Optimizer.add_to_archive(archive, new_point)
                    current_point = new_point
                    if len(archive) > soft_limit:
                        archive = Optimizer.clustering(archive, problem, hard_limit, clustering_max_iterations, print_allowed)
            else:
                raise RuntimeError(f"Something went wrong\narchive: {archive}\nx:{current_point}\ny: {new_point}\n x < y: {Optimizer.dominates(current_point, new_point)}\n y < x: {Optimizer.dominates(new_point, current_point)}\ny domination rank: {k_s_dominated_by_y}\narchive domination rank: {k_s_dominating_y}")
        return Optimizer.clustering(archive, problem, hard_limit, clustering_max_iterations, print_allowed) if clustering_before_return else archive

    @staticmethod
    def print_header(problem):
        if problem.num_of_constraints == 0:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>10} | {:>10} | {:>10} |".format("temp.", "# eval", " # nds", "D*", "Dnad", "phi"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 10, "-" * 10, "-" * 10))
        else:
            tqdm.write("\n  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))
            tqdm.write("  | {:>12} | {:>10} | {:>6} | {:>6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |".format("temp.", "# eval", "# nds", "# feas", "cv min", "cv avg", "D*", "Dnad", "phi"))
            tqdm.write("  +-{:>12}-+-{:>10}-+-{:>6}-+-{:>6}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+-{:>10}-+".format("-" * 12, "-" * 10, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10))

    def compute_deltas(self):
        objectives = np.array([s["f"] for s in self.archive])
        nadir = np.nanmax(objectives, axis = 0)
        ideal = np.nanmin(objectives, axis = 0)
        normalized_objectives = np.array([])
        try:
            normalized_objectives = np.array([[(p - i) / (n - i) for p, i, n in zip(x, ideal, nadir)] for x in objectives[:]])
            retvalue = (0, 0, 0)
            if self.nadir is not None and self.ideal is not None and self.old_norm_objectives is not None and len(self.old_norm_objectives) != 0:
                delta_nad = np.nanmax([(nad_t_1 - nad_t) / (nad_t_1 - id_t) for nad_t_1, nad_t, id_t in zip(self.nadir, nadir, ideal)])
                delta_ideal = np.nanmax([(id_t_1 - id_t) / (nad_t_1 - id_t) for id_t_1, id_t, nad_t_1 in zip(self.ideal, ideal, self.nadir)])
                phy = Optimizer.inverted_generational_distance(self.old_norm_objectives, normalized_objectives)
                self.phy.append(phy)
                retvalue = (delta_nad, delta_ideal, phy)
            self.nadir = nadir
            self.ideal = ideal
            self.old_norm_objectives = normalized_objectives
            return retvalue
        except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
            self.phy.append(0)
            return (0, 0, 0)

    def print_statistics(self, problem):
        delta_nad, delta_ideal, phy = self.compute_deltas()
        if problem.num_of_constraints == 0:
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>10.3e} | {:>10.3e} | {:>10.3e} |".format(self.current_temperature, self.n_eval, len(self.archive), delta_ideal, delta_nad, phy))
        else:
            feasible, cv_min, cv_avg = Optimizer.compute_cv(self.archive)
            tqdm.write("  | {:>12.2e} | {:>10.2e} | {:>6} | {:>6} | {:>10.2e} | {:>10.2e} | {:>10.3e} | {:>10.3e} | {:>10.3e} |".format(self.current_temperature, self.n_eval, len(self.archive), feasible, cv_min, cv_avg, delta_ideal, delta_nad, phy))

    def save_checkpoint_minimize(self):
        checkpoint = {
            "n_eval": self.n_eval,
            "t": self.current_temperature,
            "ideal": self.ideal.tolist() if self.ideal is not None else "None",
            "nadir": self.nadir.tolist() if self.nadir is not None else "None",
            "norm": self.old_norm_objectives if isinstance(self.old_norm_objectives, (list, tuple)) else self.old_norm_objectives.tolist(),
            "phy": self.phy,
            "arc": self.archive
        }
        try:
            json_string = json.dumps(checkpoint)
            with open(self.config.minimize_checkpoint_file, 'w') as outfile:
                outfile.write(json_string)
        except TypeError as e:
            print(checkpoint)
            print(e)
            exit()

    def save_checkpoint_hillclimb(self, candidate_solutions):
        try:
            json_string = json.dumps(candidate_solutions)
            with open(self.config.hill_climb_checkpoint_file, 'w') as outfile:
                outfile.write(json_string)
        except TypeError as e:
            print(candidate_solutions)
            print(e)
            exit()

    def read_checkpoint_minimize(self, problem):
        print("Resuming minimize from checkpoint...")
        with open(self.config.minimize_checkpoint_file) as file:
            checkpoint = json.load(file)
        self.n_eval = int(checkpoint["n_eval"])
        self.current_temperature = float(checkpoint["t"])
        self.ideal = [float(i) for i in checkpoint["ideal"]] if checkpoint["ideal"] != "None" else None
        self.nadir = [float(i) for i in checkpoint["nadir"]] if checkpoint["nadir"] != "None" else None
        self.old_norm_objectives = checkpoint["norm"]
        self.phy = [float(i) for i in checkpoint["phy"]]
        self.archive = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in checkpoint["arc"]]

    def read_checkpoint_hill_climb(self, problem):
        print("Resuming hill-climbing from checkpoint...")
        with open(self.config.hill_climb_checkpoint_file) as file:
            checkpoint = json.load(file)
        return [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in checkpoint]


    @staticmethod
    def is_the_same(x, y):
        return x["x"] == y["x"]

    @staticmethod
    def not_the_same(x, y):
        return x["x"] != y["x"]

    @staticmethod
    def get_objectives(problem, s):
        for i, t in zip(s["x"], problem.types):
            assert isinstance(i, int if t == Type.INTEGER else float), f"Type mismatch. This decision variable is {t}, but the internal type is {type(i)}. Please repurt this bug"
        problem.total_calls += 1
        # if s["x"] is in the cache, do not call problem.evaluate, but return the cached-entry
        if problem.is_cached(s):
            s["f"] = problem.cache[problem.get_cache_key(s)]["f"]
            s["g"] = problem.cache[problem.get_cache_key(s)]["g"]
            problem.cache_hits += 1
        else:
            # if s["x"] is not in the cache, call "evaluate" and add s["x"] to the cache
            out = {"f": [0] * problem.num_of_objectives, "g": [0] * problem.num_of_constraints if problem.num_of_constraints > 0 else None}
            problem.evaluate(s["x"], out)
            s["f"] = out["f"]
            s["g"] = out["g"]
            problem.add_to_cache(s)

    @staticmethod
    def dominates(x, y):
        if x["g"] is None:
            return all(i <= j for i, j in zip(x["f"], y["f"])) and any(i < j for i, j in zip(x["f"], y["f"]))
        else:
            return Optimizer.x_is_feasible_while_y_is_nor(x, y) or Optimizer.both_infeasible_but_x_is_better(x, y) or Optimizer.both_feasible_but_x_is_better(x, y)

    @staticmethod
    def x_is_feasible_while_y_is_nor(x, y):
        return all(i <= 0 for i in x["f"]) and any(i > 0 for i in y["g"])

    @staticmethod
    def both_infeasible_but_x_is_better(x, y):
        return any(i > 0 for i in x["g"]) and any(i > 0 for i in y["g"]) and all(i <= j for i, j in zip(x["g"], y["g"])) and any(i < j for i, j in zip(x["g"], y["g"]))

    @staticmethod
    def both_feasible_but_x_is_better(x, y):
        return all(i <= 0 for i in x["g"]) and all(i <= 0 for i in y["g"]) and all(i <= j for i, j in zip(x["f"], y["f"])) and any(i < j for i, j in zip(x["f"], y["f"]))

    @staticmethod
    def lower_point(problem):
        x = {"x": problem.lower_bound, "f": [0] * problem.num_of_objectives, "g": [0] * problem.num_of_constraints if problem.num_of_constraints > 0 else None}
        Optimizer.get_objectives(problem, x)
        return x

    @staticmethod
    def upper_point(problem):
        x = {"x": [ (x - 1) if t == Type.INTEGER else (x - 2 * np.finfo(float).eps) for x, t in zip(problem.upper_bound, problem.types)], "f": [0] * problem.num_of_objectives, "g": [0] * problem.num_of_constraints if problem.num_of_constraints > 0 else None}
        Optimizer.get_objectives(problem, x)
        return x

    @staticmethod
    def random_point(problem):
        x = {"x": [lb if lb == ub else random.randrange(lb, ub) if tp == Type.INTEGER else random.uniform(lb, ub) for lb, ub, tp in zip(problem.lower_bound, problem.upper_bound, problem.types)], "f": [0] * problem.num_of_objectives, "g": [0] * problem.num_of_constraints if problem.num_of_constraints > 0 else None}
        Optimizer.get_objectives(problem, x)
        return x

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
    def compute_fitness_range(archive, current_point, new_point):
        f = [s["f"] for s in archive] + [current_point["f"], new_point["f"]]
        return np.nanmax(f, axis = 0) - np.nanmin(f, axis = 0)

    @staticmethod
    def hill_climbing(problem, x, max_iterations):
        dimention, increase = Optimizer.hill_climbing_direction(problem)
        for _ in range(max_iterations):
            y = copy.deepcopy(x)
            Optimizer.hill_climbing_adaptive_step(problem, y, dimention, increase)
            if Optimizer.dominates(y, x) and Optimizer.not_the_same(y, x):
                x = y
            else:
                dimention, increase = Optimizer.hill_climbing_direction(problem, dimention)
        return x

    @staticmethod
    def hill_climbing_direction(problem, current_dimention = None):
        if current_dimention is None:
            return random.randrange(0, problem.num_of_variables), 1 if random.random() > 0.5 else -1
        increase = (random.random() > 0.5)
        dimention = random.randrange(0, problem.num_of_variables)
        while current_dimention == dimention:
            dimention = random.randrange(0, problem.num_of_variables)
        return dimention, increase

    @staticmethod
    def impose_domain_constraints(problem, x):
        # impose constraints the hard way
        lb = np.array(problem.lower_bound)
        ub = np.array(problem.upper_bound)
        dv = np.array(x["x"])
        dv = np.where(dv < ub, dv, ub - problem.min_step)
        dv = np.where(dv >= lb, dv, lb)
        return dv.tolist()
        
    @staticmethod
    def hill_climbing_adaptive_step(problem, x, dimention, increase):
        safety_exit = problem.max_attempt # a safety-exit prevents infinite loop, using a counter variable
        while safety_exit >= 0 and problem.is_cached(x):
            safety_exit -= 1
            tp = problem.types[dimention]
            min_step = 1 if tp == Type.INTEGER else (2 * np.finfo(float).eps)
            random_function = random.randrange if tp == Type.INTEGER else random.uniform
            max_decrease = problem.lower_bound[dimention] - x["x"][dimention]
            max_increase = problem.upper_bound[dimention] - x["x"][dimention] - min_step
            step = 0
            if increase and max_increase > 0:
                step = random_function(0, max_increase) 
            elif increase == False and max_decrease < 0:
                step = random_function(max_decrease, 0)
            x["x"][dimention] += step
            x["x"] = Optimizer.impose_domain_constraints(problem, x)
        Optimizer.get_objectives(problem, x)

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
            z["x"] = Optimizer.impose_domain_constraints(problem, z)
        Optimizer.get_objectives(problem, z)
        return z

    # @staticmethod
    # def matter_temperatures(initial_temperature, final_temperature, cooling_factor):
    #     current_temperature = [initial_temperature]
    #     while current_temperature[-1] > final_temperature * cooling_factor:
    #         current_temperature.append(float(current_temperature[-1] * cooling_factor))
    #     return current_temperature

    @staticmethod
    def add_to_archive(archive, x):
        if len(archive) == 0:
            archive.append(x)
        else:
            for y in archive:
                if Optimizer.dominates(x, y):
                    archive.remove(y)
            if not any(Optimizer.dominates(y, x) or Optimizer.is_the_same(x, y) for y in archive):
                archive.append(x)

    @staticmethod
    def nondominated_merge(archives):
        nondominated_archive = []
        for archive in tqdm(archives, desc = "Merging archives: ", leave = False):
            for x in archive:
                Optimizer.add_to_archive(nondominated_archive, x)
        return nondominated_archive

    @staticmethod
    def compute_cv(archive):
        g = np.array([s["g"] for s in archive])
        feasible = np.all(np.less(g, 0), axis = 1).sum()
        g = g[np.where(g > 0)]
        return feasible, 0 if len(g) == 0 else np.nanmin(g), 0 if len(g) == 0 else np.average(g)

    @staticmethod
    def remove_infeasible(problem, archive):
        if problem.num_of_constraints > 0:
            return [s for s in archive if all(g <= 0 for g in s["g"])]
        return archive

    @staticmethod
    def remove_dominated(archive):
        nondominated_archive = []
        for x in archive:
            Optimizer.add_to_archive(nondominated_archive, x)
        return nondominated_archive

    @staticmethod
    def clustering(archive, problem, hard_limit, max_iterations, print_allowed):
        if problem.num_of_constraints == 0:
            return Optimizer.kmeans_clustering(archive, hard_limit, max_iterations, print_allowed)
        feasible = [s for s in archive if all(g <= 0 for g in s["g"])]
        unfeasible = [s for s in archive if any(g > 0 for g in s["g"])]
        if len(feasible) > hard_limit:
            return Optimizer.kmeans_clustering(feasible, hard_limit, max_iterations, print_allowed)
        elif len(feasible) < hard_limit and len(unfeasible) != 0:
            return feasible + Optimizer.kmeans_clustering(unfeasible, hard_limit - len(feasible), max_iterations, print_allowed)
        else:
            return feasible

    @staticmethod
    def centroid_of_set(input_set):
        d = np.array([np.nansum([np.nan if np.array_equal(np.array(i["x"]), np.array(j["x"])) else np.linalg.norm(np.array(i["f"]) - np.array(j["f"])) for j in input_set]) for i in input_set])
        return input_set[np.nanargmin(d)]

    @staticmethod
    def kmeans_clustering(archive, num_of_clusters, max_iterations, print_allowed):
        assert max_iterations > 0
        if 1 < num_of_clusters < len(archive):
            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            centroids = [random.choice(archive)]
            for _ in trange(num_of_clusters - 1, desc = "Centroids", leave = False) if print_allowed else range(num_of_clusters - 1):
                # Calculate normalized distances from points to the centroids
                dists = np.array([np.nansum([np.linalg.norm(np.array(centroid["f"]) - np.array(p["f"])) for centroid in centroids]) for p in archive])
                try:
                    normalized_dists = dists / np.nansum(dists)
                    # Choose remaining points based on their distances
                    new_centroid_idx = np.random.choice(range(len(archive)), size = 1, p = normalized_dists)[0]  # Indexed @ zero to get val, not array of val
                    centroids += [archive[new_centroid_idx]]
                except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
                    print(e)
                    print(f"Archive: {archive}")
                    print(f"Centroids: {centroids}")
                    print(f"Distance: {dists}")
                    print(f"Normalized distance: {dists / np.nansum(dists)}")
                    exit()
            # Iterate, adjusting centroids until converged or until passed max_iter
            for _ in trange(max_iterations, desc = "K-means", leave = False) if print_allowed else range(max_iterations):
                # Sort each datapoint, assigning to nearest centroid
                sorted_points = [[] for _ in range(num_of_clusters)]
                for x in archive:
                    dists = [np.linalg.norm(np.array(x["f"]) - np.array(centroid["f"])) for centroid in centroids]
                    centroid_idx = np.argmin(dists)
                    sorted_points[centroid_idx].append(x)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = centroids
                centroids = [Optimizer.centroid_of_set(cluster) if len(cluster) != 0 else centroid for cluster, centroid in zip(sorted_points, prev_centroids)]
                if np.array_equal(centroids, prev_centroids) and print_allowed:
                    break
            return centroids
        elif num_of_clusters == 1:
            return [Optimizer.centroid_of_set(archive)]
        else:
            return archive

    @staticmethod
    def inverted_generational_distance(p_t, p_tau):
        return np.nansum([np.nanmin([np.linalg.norm(p - q) for q in p_t[:]]) for p in p_tau[:]]) / len(p_tau)
