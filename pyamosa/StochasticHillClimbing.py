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
import json5, random, numpy as np, copy
from .Problem import Problem
from .Pareto import Pareto
from .DataType import Type
from tqdm import tqdm, trange

class StochasticHillClimbing:
    
    def __init__(self, problem : Problem, pareto: Pareto, checkpoint_file : str = ".hill_climb_checkpoint.json5") -> None:
        self.problem = problem
        self.pareto = pareto
        self.checkpoint_file = checkpoint_file
        
    def init(self):
        self.pareto.candidate_solutions = [ self.problem.lower_point(), self.problem.upper_point() ]
        
    def run(self, max_num_of_candidates, max_iterations):
        for _ in trange(len(self.pareto.candidate_solutions), max_num_of_candidates, desc = "Generating initial candidates: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            self.climb(self.problem.random_point(), max_iterations)
            self.save_checkpoint()

    def climb(self, candidate, max_iterations):
        direction, heading = self.stochastic_steep()
        step_size = self.min_step(direction)
        for _ in trange(max_iterations, desc = "Hill climbing: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            new_candidate = copy.deepcopy(candidate)
            new_candidate["x"][direction] =  StochasticHillClimbing.clip(new_candidate["x"][direction] + (step_size * heading), self.problem.lower_bound[direction], self.problem.upper_bound[direction] - self.min_step(direction))
            assert self.problem.lower_bound[direction] <= new_candidate["x"][direction] <= self.problem.upper_bound[direction], f"Variable {direction} with value {new_candidate['x'][direction]} is out of bound for [{self.problem.lower_bound[direction]}, {self.problem.upper_bound[direction]}]"
            self.problem.get_objectives(new_candidate)
            if Pareto.dominates(new_candidate, candidate) or (not Pareto.dominates(new_candidate, candidate) and not Pareto.dominates(candidate, new_candidate) and Pareto.not_the_same(candidate, new_candidate)):
                candidate = new_candidate
                step_size = StochasticHillClimbing.clip(step_size * 2, self.min_step(direction), self.problem.upper_bound[direction] - self.min_step(direction) - new_candidate["x"][direction] if heading == 1 else new_candidate["x"][direction] - self.problem.lower_bound[direction])
            else:
                direction, heading = self.stochastic_steep()
                step_size = self.min_step(direction)
        self.pareto.candidate_solutions.append(candidate)

    def min_step(self, direction):
        return 1 if self.problem.types[direction] == Type.INTEGER else (5 * np.finfo(float).eps)
    
    def stochastic_steep(self):
        direction = random.randrange(0, self.problem.num_of_variables)
        while self.problem.lower_bound[direction] == self.problem.upper_bound[direction]:
            direction = random.randrange(0, self.problem.num_of_variables)
        heading = 1 if random.random() > 0.5 else -1
        return direction, heading
    
    @staticmethod
    def clip(x, xmin, xmax):
        #assert xmin < xmax, f"xmin: {xmin}, xmax: {xmax}"
        return max(min(x, xmax), xmin)
    
    def save_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'w') as outfile:
                json5.dump(self.pareto.candidate_solutions, outfile)
        except TypeError as e:
            print(self.pareto.candidate_solutions)
            print(e)
            exit()
            
    def read_checkpoint(self):
        with open(self.checkpoint_file) as file:
            checkpoint = json5.load(file)
        self.pareto.candidate_solutions = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], self.problem.types)], "f": a["f"], "g": a["g"]} for a in checkpoint]




