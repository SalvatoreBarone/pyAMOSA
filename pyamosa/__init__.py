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
from .DataType import Type
from .Problem import Problem
from .Config import Config
from .Optimizer import Optimizer 
from .StopCriterion import StopCriterion
from .StopMaxTime import StopMaxTime
from .StopMinTemperature import StopMinTemperature
from .StopPhyWindow import StopPhyWindow

name = "pyamosa"
__version__ = "1.2.3"
__author__ = "Salvatore Barone"
__credits__ = "Department of Electrical Engineering and Information Technologies, University of Naples Federico II, Via Claudio 21, Naples, Italy"