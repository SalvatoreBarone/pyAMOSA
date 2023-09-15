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

class HillClimbing:
    
    def __init__(self, problem : Problem, pareto: Pareto, checkpoint_file : str = "hill_climb_checkpoint.json5") -> None:
        self.problem = problem
        self.pareto = pareto
        self.checkpoint_file = checkpoint_file
        
    def init(self):
        self.pareto.candidate_solutions = [ self.problem.lower_point(), self.problem.upper_point() ]
        
    def run(self, max_num_of_candidates, max_iterations):
        for _ in trange(len(self.pareto.candidate_solutions), max_num_of_candidates, desc = "Hill climbing", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            dimention, increase = self.direction(self.problem)
            x = self.problem.random_point()
            for _ in trange(max_iterations, desc = "Walking...", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
                y = copy.deepcopy(x)
                self.step(y, dimention, increase)
                if Pareto.dominates(y, x) and Pareto.not_the_same(y, x):
                    x = y
                else:
                    dimention, increase = self.direction(dimention)
            self.pareto.candidate_solutions.append(x)
            self.save_checkpoint()
    
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

    def direction(self, current_dimention = None):
        if current_dimention is None:
            return random.randrange(0, self.problem.num_of_variables), 1 if random.random() > 0.5 else -1
        increase = (random.random() > 0.5)
        dimention = random.randrange(0, self.problem.num_of_variables)
        while current_dimention == dimention:
            dimention = random.randrange(0, self.problem.num_of_variables)
        return dimention, increase

    def step(self, x, dimention, increase):
        safety_exit = self.problem.max_attempt # a safety-exit prevents infinite loop, using a counter variable
        while safety_exit >= 0 and self.problem.is_cached(x):
            safety_exit -= 1
            tp = self.problem.types[dimention]
            min_step = 1 if tp == Type.INTEGER else (2 * np.finfo(float).eps)
            random_function = random.randrange if tp == Type.INTEGER else random.uniform
            max_decrease = self.problem.lower_bound[dimention] - x["x"][dimention]
            max_increase = self.problem.upper_bound[dimention] - x["x"][dimention] - min_step
            step = 0
            if increase and max_increase > 0:
                step = random_function(0, max_increase) 
            elif increase == False and max_decrease < 0:
                step = random_function(max_decrease, 0)
            x["x"][dimention] += step
        self.problem.get_objectives(x)



