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
import numpy as np, networkx as nx, json5
from enum import Enum
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from .DataType import Type
from .Problem import Problem

class VariableGrouping:

    """
    For firther details on Transfer Strategies (TSs), please refer to

    Sander, Frederick, Heiner Zille, e Sanaz Mostaghim. "Transfer Strategies from Single- to Multi-Objective
    Grouping Mechanisms". In Proceedings of the Genetic and Evolutionary Computation Conference, 729-36.
    Kyoto Japan: ACM, 2018. https://doi.org/10.1145/3205455.3205491.

    """
    class TSObjective(Enum):
        Any = 1 # regards two variables xi and xj as interacting, if an interaction exists in *any* of the objective functions
        All = 2 # regards two variables xi and xj as interacting, if an interaction exists in *all* of the objective functions

    class TSVariable(Enum):
        Any = 1 # A variable xi is added to a group if the combined interaction graph contains an edge between xi and any variable xj in the group.
        All = 2 # A variable xi is added to a group if the combined interaction graph contains an edge between xi and all the other variable xj in the group.

    def __init__(self, problem : Problem):
        self.problem = problem
        self.variable_masks = []

    def load(self, cache : str = "grouping_cache.json5"):
        with open(cache, "r") as file:
            self.variable_masks = json5.load(file)

    def store(self, cache : str = "grouping_cache.json5"):
        try:
            with open(cache, 'w') as outfile:
                json5.dump(self.variable_masks, outfile)
        except TypeError as e:
            print(self.variable_masks)
            print(e)
            exit()

    def min_step(self, dimension):
        return 1 if self.problem.types[dimension] == Type.INTEGER else (5 * np.finfo(float).eps)

    def run(self, tso : TSObjective = TSObjective.Any, tsv : TSVariable = TSVariable.Any):
        pass

    def print_masks(self):
        for i, g in enumerate(self.variable_masks):
            print(f"Group {i} of size {np.sum(g)}: {g}")


    def TSV_all(self, Theta):
        interacting_variables = []
        graph = nx.from_numpy_array(Theta)
        cliques = [s for s in nx.find_cliques(graph) if len(s) > 1]
        for indices in cliques:
            if all(i not in g for g in interacting_variables for i in indices):
                interacting_variables.append(indices)
        separable_variables = [i for i in range(np.shape(Theta)[0]) if all(i not in g for g in interacting_variables) ]   
        self.gen_masks(Theta, interacting_variables, separable_variables)

    def TSV_any(self, Theta):
        adjiacente_list = { i : np.where(Theta[i] != 0)[0].tolist() for i in range(np.shape(Theta)[0])}
        interacting_variables = []
        separable_variables = []
        for dv, interactions in adjiacente_list.items():
            if len(interactions) > 1 and all(dv not in g for g in interacting_variables):
                interacting_variables.append([dv] + interactions)
            elif len(interactions) == 0 or (len(interactions) == 1 and all(i not in g for g in interacting_variables for i in interactions)):
                separable_variables.append(dv)
        self.gen_masks(Theta, interacting_variables, separable_variables)
        
    def gen_masks(self, Theta, interacting_variables, separable_variables):
        self.variable_masks = []
        for i in interacting_variables:
            mask = np.zeros((np.shape(Theta)[0],))
            np.put_along_axis(mask, np.array(i), values = [1], axis = 0)
            self.variable_masks.append(mask.tolist())
        if separable_variables:
            mask = np.zeros((np.shape(Theta)[0],))
            np.put_along_axis(mask, np.array(separable_variables), values = [1], axis = 0)
            self.variable_masks.append(mask.tolist())
