"""
Graphem: A graph embedding library based on JAX for efficient k-nearest neighbors.
"""

from graphem.embedder import GraphEmbedder
from graphem.index import HPIndex
from graphem.influence import graphem_seed_selection, ndlib_estimated_influence, greedy_seed_selection
from graphem.datasets import load_dataset
from graphem.generators import (
    generate_erdos_renyi_graph,
    generate_barabasi_albert_graph,
    generate_watts_strogatz_graph,
    generate_random_regular_graph,
    generate_sbm_graph,
    generate_scale_free_graph,
    generate_geometric_graph,
    generate_caveman_graph,
    generate_relaxed_caveman_graph
)
from graphem.visualization import visualize_graph, plot_benchmark_results

__version__ = '0.1.0'
