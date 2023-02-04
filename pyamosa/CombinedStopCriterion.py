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
from .StopMaxTime import StopMaxTime
from .StopMinTemperature import StopMinTemperature
from .StopPhyWindow import StopPhyWindow

class CombinedStopCriterion():
    def __init__(self, max_duration : str, min_temperature : float, termination_window: int):
        self.max_py = StopPhyWindow(termination_window)
        self.min_temperat = StopMinTemperature(min_temperature)
        self.max_duration = StopMaxTime(max_duration)

    def check_termination(self, optimizer):
        return self.max_duration.check_termination(optimizer) or self.min_temperat.check_termination(optimizer) or self.max_py.check_termination(optimizer)

    def info(self):
        self.max_py.info()
        self.max_duration.info()
        self.min_temperat.info()
        
        
