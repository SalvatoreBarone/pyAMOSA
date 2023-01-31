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
import time, datetime
from .StopCriterion import StopCriterion


class StopMaxTime(StopCriterion):
    def __init__(self, max_duration : str):
        self.max_seconds = sum([int(i) * j for i,j in zip(max_duration.split(':'), [3600, 60, 1])])     

    def check_termination(self, optimizer):
        return (time.time() - optimizer.duration > self.max_seconds)

    def info(self):
        print(f"Maximum duration: {str(datetime.timedelta(seconds=self.max_seconds))}")