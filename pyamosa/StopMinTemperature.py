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


class StopMinTemperature(StopCriterion):
    def __init__(self, min_temperature : float):
        assert min_temperature > 0
        self.min_temperature = float(min_temperature)
        pass

    def check_termination(self, optimizer):
        return (optimizer.current_temperature < self.min_temperature)

    def info(self):
        print(f"Minumum temperature: {self.min_temperature} degrees")