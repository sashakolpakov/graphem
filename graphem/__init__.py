"""
Graphem: A graph embedding library based on JAX for efficient k-nearest neighbors.
"""

from graphem.embedder import GraphEmbedder
from graphem.index import HPIndex
from graphem.influence import graphem_seed_selection, ndlib_estimated_influence, greedy_seed_selection
from graphem.datasets import load_dataset
from graphem.generators import (
    erdos_renyi_graph,
    generate_ba,
    generate_ws,
    generate_random_regular,
    generate_sbm,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_relaxed_caveman
)
from graphem.visualization import visualize_graph, plot_benchmark_results

__version__ = '0.1.0'
