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
from .StopCriterion import StopCriterion
import numpy as np
class StopPhyWindow(StopCriterion):
    def __init__(self, termination_window : int):
        assert termination_window > 0
        self.window_width = int(termination_window)

    def check_termination(self, optimizer):
        return (len(optimizer.phy) > self.window_width and all(optimizer.phy[-self.window_width:] <= np.finfo(float).eps))

    def info(self):
        print(f"IGD window width: {self.window_width}")