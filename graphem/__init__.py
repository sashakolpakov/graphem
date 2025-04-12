"""
Graphem: A graph embedding library based on JAX for efficient k-nearest neighbors.
"""

from graphem.embedder import GraphEmbedder
from graphem.index import HPIndex
from graphem.influence import graphem_seed_selection, ndlib_estimated_influence, greedy_seed_selection

__version__ = '0.1.0'
