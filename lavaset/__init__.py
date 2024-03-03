from pkg_resources import get_distribution
__version__ = get_distribution('lavaset').version

from .lavaset import LAVASET
from .rf import StochasticBosque
# from .best_cut_node import best_cut_node