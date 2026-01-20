"""DLA (Diffusion-Limited Aggregation) Solver - Idealized Version"""

__version__ = "0.0.4"
__author__ = "Sandy H. S. Herho, Faiz R. Fajary, Iwan P. Anwar, Faruq Khadami, Nurjanna J. Trilaksono, Rusmawan Suwarman, Dasapta E. Irawan"

from .core.solver import DLASolver
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = ["DLASolver", "ConfigManager", "DataHandler"]
