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

class Config:
    def __init__(
            self,
            archive_hard_limit : int = 20,
            archive_soft_limit : int = 50,
            archive_gamma : int = 2,
            clustering_max_iterations : int = 300,
            hill_climbing_iterations : int = 500,
            initial_temperature : int = 500,
            cooling_factor : float = 0.9,
            annealing_iterations : int = 500,
            annealing_strength : int = 1,
            multiprocessing_enabled : bool = True,
            hill_climb_checkpoint_file : str = "hill_climb_checkpoint.json",
            minimize_checkpoint_file : str = "minimize_checkpoint.json",
            cache_dir :str = ".cache"
            ):

        assert archive_soft_limit >= archive_hard_limit > 0, f"soft limit: {archive_soft_limit}, hard limit: {archive_hard_limit}"
        assert archive_gamma > 0, f"gamma: {archive_gamma}"
        assert clustering_max_iterations > 0, f"clustering iterations: {clustering_max_iterations}"
        assert hill_climbing_iterations >= 0, f"hill-climbing iterations: {hill_climbing_iterations}"
        assert 0 < cooling_factor < 1, f"cooling factor: {cooling_factor}"
        assert annealing_iterations > 0, f"annealing iterations: {annealing_strength}"
        assert annealing_strength >= 1, f"annealing strength: {annealing_strength}"
        self.archive_hard_limit = archive_hard_limit
        self.archive_soft_limit = archive_soft_limit
        self.clustering_max_iterations = clustering_max_iterations
        self.archive_gamma = archive_gamma
        self.hill_climbing_iterations = hill_climbing_iterations
        self.initial_temperature = initial_temperature
        self.cooling_factor = cooling_factor
        self.annealing_iterations = annealing_iterations
        self.annealing_strength = annealing_strength
        self.multiprocessing_enabled = multiprocessing_enabled
        self.hill_climb_checkpoint_file = hill_climb_checkpoint_file
        self.minimize_checkpoint_file = minimize_checkpoint_file
        self.cache_dir = cache_dir