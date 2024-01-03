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
from __future__ import annotations
import numpy as np, matplotlib.pyplot as plt, json5, sys, os, random
from tqdm import tqdm, trange
from .DataType import Type
from .Problem import Problem
class Pareto:
    def __init__(self) -> None:
        self.candidate_solutions = []
        self.ideal = None
        self.nadir = None
        self.C_actual_prev = None
        self.old_norm_objectives = []
        self.phy = []
        
    @staticmethod
    def is_the_same(x, y):
        return x["x"] == y["x"]

    @staticmethod
    def not_the_same(x, y):
        return x["x"] != y["x"]

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
    def dominates(x, y):
        if x["g"] is None:
            return all(i <= j for i, j in zip(x["f"], y["f"])) and any(i < j for i, j in zip(x["f"], y["f"]))
        else:
            return Pareto.x_is_feasible_while_y_is_nor(x, y) or Pareto.both_infeasible_but_x_is_better(x, y) or Pareto.both_feasible_but_x_is_better(x, y)

    @staticmethod
    def coverage(A, B):
        if len(B) == 0 or len(A) == 0:
            return 0
        assert A.shape[1] == B.shape[1]
        count = 0
        for b in B:
            for a in A:
                if all(i <= j for i, j in zip(a, b)) and any(i < j for i, j  in zip(a, b)):
                    count += 1 
                    break
        return count / B.shape[0]

    @staticmethod
    def inverted_generational_distance(reference, evolved):
        return np.nansum([np.nanmin([np.linalg.norm(p - q) for q in reference[:]]) for p in evolved[:]]) / len(evolved)
        
    @staticmethod
    def centroid_of_set(input_set):
        d = np.array([np.nansum([np.nan if np.array_equal(np.array(i["x"]), np.array(j["x"])) else np.linalg.norm(np.array(i["f"]) - np.array(j["f"])) for j in input_set]) for i in input_set])
        return input_set[np.nanargmin(d)]
    
    @staticmethod
    def kmeans_clustering(set_of_points, num_of_clusters, max_iterations):
        assert max_iterations > 0
        if 1 < num_of_clusters < len(set_of_points):
            # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
            # then the rest are initialized w/ probabilities proportional to their distances to the first
            # Pick a random point from train data for first centroid
            centroids = [random.choice(set_of_points)]
            for _ in trange(num_of_clusters - 1, desc = "Centroids", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
                # Calculate normalized distances from points to the centroids
                dists = np.array([np.nansum([np.linalg.norm(np.array(centroid["f"]) - np.array(p["f"])) for centroid in centroids]) for p in set_of_points])
                try:
                    normalized_dists = dists / np.nansum(dists)
                    # Choose remaining points based on their distances
                    new_centroid_idx = np.random.choice(range(len(set_of_points)), size = 1, p = normalized_dists)[0]  # Indexed @ zero to get val, not array of val
                    centroids += [set_of_points[new_centroid_idx]]
                except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
                    print(e)
                    # print(f"Archive: {set_of_points}")
                    # print(f"Centroids: {centroids}")
                    # print(f"Distance: {dists}")
                    # print(f"Normalized distance: {dists / np.nansum(dists)}")
                    exit()
            # Iterate, adjusting centroids until converged or until passed max_iter
            for _ in trange(max_iterations, desc = "K-means", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
                # Sort each datapoint, assigning to nearest centroid
                sorted_points = [[] for _ in range(num_of_clusters)]
                for x in set_of_points:
                    dists = [np.linalg.norm(np.array(x["f"]) - np.array(centroid["f"])) for centroid in centroids]
                    centroid_idx = np.argmin(dists)
                    sorted_points[centroid_idx].append(x)
                # Push current centroids to previous, reassign centroids as mean of the points belonging to them
                prev_centroids = centroids
                centroids = [Pareto.centroid_of_set(cluster) if len(cluster) != 0 else centroid for cluster, centroid in zip(sorted_points, prev_centroids)]
                if np.array_equal(centroids, prev_centroids):
                    break
            return centroids
        elif num_of_clusters == 1:
            return [Pareto.centroid_of_set(set_of_points)]
        else:
            return set_of_points

    def write_json(self, json_file : str):
        try:
            with open(json_file, 'w') as outfile:
                json5.dump(self.candidate_solutions, outfile)
        except TypeError as e:
            print(self.candidate_solutions)
            print(e)
            exit()
            
    def read_json(self, problem : Problem, json_file : str):
        with open(json_file) as f:
            archive = json5.load(f)
        self.candidate_solutions = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in archive]
        
    def get_checkpoint(self):
        return {
            "ideal": self.ideal.tolist() if self.ideal is not None else "None",
            "nadir": self.nadir.tolist() if self.nadir is not None else "None",
            "norm": self.old_norm_objectives if isinstance(self.old_norm_objectives, (list, tuple)) else self.old_norm_objectives.tolist(),
            "phy": self.phy,
            "arc": self.candidate_solutions}
    
    def from_checkpoint(self, checkpoint, problem : Problem):
        self.ideal = [float(i) for i in checkpoint["ideal"]] if checkpoint["ideal"] != "None" else None
        self.nadir = [float(i) for i in checkpoint["nadir"]] if checkpoint["nadir"] != "None" else None
        self.old_norm_objectives = np.array(checkpoint["norm"])
        self.phy = [float(i) for i in checkpoint["phy"]]
        self.candidate_solutions = [{"x": [int(i) if j == Type.INTEGER else float(i) for i, j in zip(a["x"], problem.types)], "f": a["f"], "g": a["g"]} for a in checkpoint["arc"]]   
        
    def export_csv(self, problem : Problem, csv_file : str, fitness_labels : list = None):
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
        
    def add(self, x):
        if len(self.candidate_solutions) == 0:
            self.candidate_solutions.append(x)
        else:
            for y in self.candidate_solutions:
                if Pareto.dominates(x, y):
                    self.candidate_solutions.remove(y)
            if not any(Pareto.dominates(y, x) or Pareto.is_the_same(x, y) for y in self.candidate_solutions):
                self.candidate_solutions.append(x)
                
    def size(self):
        return len(self.candidate_solutions)
    
    def random_point(self):
        return random.choice(self.candidate_solutions)
                
    def merge(self, paretos : list):
        for pareto in tqdm(paretos, desc = "Merging archives: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}"):
            for x in pareto:
                self.add(x)
                
    def clustering(self, problem : Problem, max_points, max_iterations):
        if problem.num_of_constraints == 0:
            self.candidate_solutions = Pareto.kmeans_clustering(self.candidate_solutions, max_points, max_iterations)
        else:
            feasible = [s for s in self.candidate_solutions if all(g <= 0 for g in s["g"])]
            unfeasible = [s for s in self.candidate_solutions if any(g > 0 for g in s["g"])]
            if len(feasible) > max_points:
                self.candidate_solutions = Pareto.kmeans_clustering(feasible, max_points, max_iterations)
            elif len(feasible) < max_points and len(unfeasible) != 0:
                self.candidate_solutions = feasible + Pareto.kmeans_clustering(unfeasible, max_points - len(feasible), max_iterations)
            else:
                self.candidate_solutions = feasible
    
    def remove_infeasible(self, problem : Problem):
        if problem.num_of_constraints > 0:
            self.candidate_solutions = [s for s in self.candidate_solutions if all(g <= 0 for g in s["g"])]
        
    def remove_dominated(self):
        self.candidate_solutions = [x for x in self.candidate_solutions if not any(Pareto.dominates(y, x) for y in self.candidate_solutions if not Pareto.is_the_same(x, y))]
        
    def dominated_by(self, s):
        return [x for x in self.candidate_solutions if Pareto.dominates(s, x)]
        
    def dominating(self, s):
        return [x for x in self.candidate_solutions if Pareto.dominates(x, s)]
    
    def get_min_agv_cv(self):
        g = np.array([s["g"] for s in self.candidate_solutions])
        feasible = np.all(np.less(g, 0), axis = 1).sum()
        g = g[np.where(g > 0)]
        return feasible, 0 if len(g) == 0 else np.nanmin(g), 0 if len(g) == 0 else np.average(g)
    
    def get_front(self):
        return np.array([s["f"] for s in self.candidate_solutions])

    def get_set(self):
        return np.array([s["x"] for s in self.candidate_solutions])

    def get_cv(self):
        return np.array([s["g"] for s in self.candidate_solutions])
    
    def compute_fitness_range(self, additional_points = None):
        f = [s["f"] for s in self.candidate_solutions] + [x["f"] for x in additional_points if additional_points is not None]
        return np.nanmax(f, axis = 0) - np.nanmin(f, axis = 0)
    
    def compute_deltas(self):
        objectives = np.array([s["f"] for s in self.candidate_solutions])
        nadir = np.nanmax(objectives, axis = 0)
        ideal = np.nanmin(objectives, axis = 0)
        C_actual_prev = 0
        normalized_objectives = np.array([])
        try:
            normalized_objectives = np.array([[(p - i) / (n - i) for p, i, n in zip(x, ideal, nadir)] for x in objectives[:]])
            retvalue = (0, 0, 0, 0, 0)
            if self.nadir is not None and self.ideal is not None and self.old_norm_objectives is not None and len(self.old_norm_objectives) != 0:
                delta_nad = np.nanmax([(nad_t_1 - nad_t) / (nad_t_1 - id_t) for nad_t_1, nad_t, id_t in zip(self.nadir, nadir, ideal)])
                delta_ideal = np.nanmax([(id_t_1 - id_t) / (nad_t_1 - id_t) for id_t_1, id_t, nad_t_1 in zip(self.ideal, ideal, self.nadir)])
                phy = Pareto.inverted_generational_distance(self.old_norm_objectives, normalized_objectives)
                self.phy.append(phy)
                C_prev_actual = Pareto.coverage(self.old_norm_objectives, normalized_objectives)
                C_actual_prev = Pareto.coverage(normalized_objectives, self.old_norm_objectives)
                retvalue = (delta_nad, delta_ideal, phy, C_prev_actual, C_actual_prev)
            self.nadir = nadir
            self.ideal = ideal
            self.C_actual_prev = C_actual_prev
            self.old_norm_objectives = normalized_objectives
            return retvalue
        except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
            self.phy.append(0)
            return (0, 0, 0, 0, 0)
    
    def plot_front(self, problem : Problem, pdf_file : str, fig_title : str = "Pareto front", axis_labels : list = None, color = "k", marker = "."):
        def draw_proj(axis, data_x, data_y, data_z, color, ranges = [0.1, 0.1, 0.1]):
            xlim = axis.get_xlim()
            ylim = axis.get_ylim()
            zlim = axis.get_zlim()
            for x, y, z in zip (data_x, data_y, data_z):
                line_x = np.arange(x, xlim[1], ranges[0])
                line_y = np.arange(ylim[0], y, ranges[1])
                line_z = np.arange(zlim[0], z, ranges[2])
                axis.plot(line_x, np.full(np.shape(line_x), y), np.full(np.shape(line_x), zlim[0]), f":{color}")
                axis.plot(np.full(np.shape(line_y), x), line_y, np.full(np.shape(line_y), zlim[0]), f":{color}")
                axis.plot(np.full(np.shape(line_z), x), np.full(np.shape(line_z), y), line_z,       f":{color}")
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_zlim(zlim)
        if axis_labels is None:
            axis_labels = [f"f{str(i)}" for i in range(problem.num_of_objectives)]
        F = self.get_front()
        if problem.num_of_objectives == 2:
            plt.figure(figsize = (10, 10), dpi = 300)
            plt.plot(F[:, 0], F[:, 1], f'{color}{marker}')
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            plt.title(fig_title)
            plt.savefig(pdf_file, bbox_inches = 'tight', pad_inches = 0)
        elif problem.num_of_objectives == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], marker = marker, color = color, depthshade = False)
            draw_proj(ax, F[:, 0], F[:, 1], F[:, 2], color = color)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            ax.set_proj_type('ortho')
            plt.title(fig_title)
            plt.tight_layout()
            plt.savefig(pdf_file, bbox_inches = 'tight', pad_inches = 0.5)
