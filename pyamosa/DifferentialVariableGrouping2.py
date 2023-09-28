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
from .DataType import Type
from .Problem import Problem
from .VariableGrouping import VariableGrouping

"""
This optimizer class is intended to be used to tacke large scale oprimization problems.
For further insights, please refer to

    Omidvar, Mohammad Nabi, Ming Yang, Yi Mei, Xiaodong Li, e Xin Yao. "DG2: A Faster and More Accurate Differential 
    Grouping for Large-Scale Black-Box Optimization". IEEE Transactions on Evolutionary Computation 21, fasc. 6
    pages 929–42 2016 https://doi.org/10.1109/TEVC.2017.2694221.

"""
class DifferentialVariableGrouping2(VariableGrouping):
    def __init__(self, problem : Problem):
        super().__init__(problem)
        self.Lambda = np.zeros((problem.num_of_variables, problem.num_of_variables, problem.num_of_objectives))
        self.F = np.empty((problem.num_of_variables, problem.num_of_variables, problem.num_of_objectives))
        self.F.fill(np.nan)
        self.f = np.empty((problem.num_of_variables, problem.num_of_objectives))
        self.f.fill(np.nan)
        self.x_1 = self.problem.lower_point()
        self.m = [ (u + l) / 2 if t == Type.REAL else (u + l) // 2 for u, l, t in zip(problem.upper_bound, problem.lower_bound, problem.types) ]
        self.Theta = np.empty((problem.num_of_variables, problem.num_of_variables))
        self.Theta.fill(np.nan)
        self.eta_0 = 0
        self.eta_1 = 0

    def reset(self):
        self.Lambda = np.zeros((self.problem.num_of_variables, self.problem.num_of_variables, self.problem.num_of_objectives))
        self.F = np.empty((self.problem.num_of_variables, self.problem.num_of_variables, self.problem.num_of_objectives))
        self.F.fill(np.nan)
        self.f = np.empty((self.problem.num_of_variables, self.problem.num_of_objectives))
        self.f.fill(np.nan)
        self.x_1 = self.problem.lower_point()
        self.m = [ (u + l) / 2 if t == Type.REAL else (u + l) // 2 for u, l, t in zip(self.problem.upper_bound, self.problem.lower_bound, self.problem.types) ]
        self.Theta = np.empty((self.problem.num_of_variables, self.problem.num_of_variables))
        self.Theta.fill(np.nan)
        self.eta_0 = 0
        self.eta_1 = 0

    def compute_ism(self):
        for i in trange(self.problem.num_of_variables, desc = "Building Lambda: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            if all(np.isnan(self.f[i])):
                x_2 = copy.deepcopy(self.x_1)
                x_2["x"][i] = self.m[i]
                self.problem.get_objectives(x_2)
                self.f[i] = x_2["f"]
            for j in trange(i + 1, self.problem.num_of_variables, desc = f"Interdipendences of x_{i}: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
                if all(np.isnan(self.f[j])):
                    x_3 = copy.deepcopy(self.x_1) 
                    x_3["x"][j] = self.m[j]
                    self.problem.get_objectives(x_3)
                    self.f[j] = x_3["f"]
                x_4 = copy.deepcopy(self.x_1) 
                x_4["x"][i] = self.m[i]
                x_4["x"][j] = self.m[j]
                self.problem.get_objectives(x_4)
                self.F[i][j] = x_4["f"]
                delta_1 = self.f[i] - self.x_1["f"]
                delta_2 = self.F[i][j] - self.f[j]
                self.Lambda[i][j] = np.absolute(delta_1 - delta_2)

    def compute_dsm(self, tso : VariableGrouping.TSObjective = VariableGrouping.TSObjective.Any):
        aggregator = {VariableGrouping.TSObjective.Any : any, VariableGrouping.TSObjective.All : all}
        for i in trange(self.problem.num_of_variables,  desc = "Building Theta: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            for j in range(i + 1, self.problem.num_of_variables):
                f_values = [self.x_1["f"], self.F[i][j], self.f[i], self.f[j]]
                e_inf = self.gamma(2) * np.sum(np.absolute(f_values))
                e_sup = self.gamma(np.sqrt(self.problem.num_of_variables)) * np.max(f_values)
                if aggregator[tso](self.Lambda[i][j] > e_sup):
                    self.Theta[i][j] = self.Theta[j][i] = 1
                    self.eta_1 += 1
                elif aggregator[tso](self.Lambda[i][j] < e_inf):
                    self.Theta[i][j] = self.Theta[j][i] = 0
                    self.eta_0 += 1
        for i in trange(self.problem.num_of_variables,  desc = "Completing Theta: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            for j in range(i + 1, self.problem.num_of_variables):
                if np.isnan(self.Theta[i][j]):
                    f_values = [self.x_1["f"], self.F[i][j], self.f[i], self.f[j]]
                    e_inf = self.gamma(2) * np.sum(np.absolute(f_values))
                    e_sup = self.gamma(np.sqrt(self.problem.num_of_variables)) * np.max(f_values)
                    eps = (self.eta_0 * e_inf + self.eta_1 * e_sup) / (self.eta_1 + self.eta_0)
                    self.Theta[i][j] = self.Theta[j][i] = aggregator[tso](self.Lambda[i][j] > eps)
        np.fill_diagonal(self.Theta, 0)
        np.nan_to_num(self.Theta, copy=False, nan=0.0)

    def compute_cc(self, tsv : VariableGrouping.TSVariable = VariableGrouping.TSVariable.Any):
        selector = {VariableGrouping.TSVariable.Any : self.TSV_any, VariableGrouping.TSVariable.All : self.TSV_all}
        selector[tsv](self.Theta)
        
    def print_theta(self):
        print("\nTheta")
        for i in range(self.problem.num_of_variables):
            for j in range(self.problem.num_of_variables):
                print("{:>10.2e}".format(self.Theta[i][j]), end = "\t")
            print("")

    def print_lambda(self):
        print("\nLambda")
        for k in range(self.problem.num_of_objectives):
            print(f"\nk = {k}")
            for i in range(self.problem.num_of_variables):
                for j in range(self.problem.num_of_variables):
                    print("{:>10.2e}".format(self.Lambda[i][j][k]), end = "\t")
                print("")

    def run(self, tso : VariableGrouping.TSObjective = VariableGrouping.TSObjective.Any, tsv : VariableGrouping.TSVariable = VariableGrouping.TSVariable.Any):
        self.reset()
        self.compute_ism()
        self.compute_dsm(tso)
        self.compute_cc(tsv)

    @staticmethod
    def gamma(k: float): # see Remark 1.2 of R. M. Corless and N. Fillion, "A Graduate Introduction to Numerical Methods". New York, NY, USA: Springer-Verlag, 2013.
        mu_m = np.finfo(float).eps
        return k * mu_m / (1 - k * mu_m)
    


