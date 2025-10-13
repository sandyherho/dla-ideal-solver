"""DLA (Diffusion-Limited Aggregation) Solver - Idealized Version"""

__version__ = "0.0.2"
__author__ = "Sandy H. S. Herho, Siti N. Kaban, Nurjanna J. Trilaksono, Iwan P. Anwar, Rusmawan Suwarman, Dasapta E. Irawan"

from .core.solver import DLASolver
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = ["DLASolver", "ConfigManager", "DataHandler"]
