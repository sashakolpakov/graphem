"""
Graphem: A graph embedding library based on JAX for efficient k-nearest neighbors.
"""

from graphem.embedder import GraphEmbedder
from graphem.index import HPIndex
from graphem.influence import graphem_seed_selection, ndlib_estimated_influence, greedy_seed_selection
from graphem.datasets import load_dataset
from graphem.generators import (
    erdos_renyi_graph,
    generate_sbm,
    generate_ba,
    generate_ws,
    generate_caveman,
    generate_geometric,
    generate_scale_free,
    generate_road_network,
    generate_balanced_tree,
    generate_power_cluster,
    generate_random_regular,
    generate_bipartite_graph,
    generate_relaxed_caveman
)
from graphem.visualization import (
    report_corr,
    report_full_correlation_matrix,
    plot_radial_vs_centrality,
    display_benchmark_results
)

__version__ = '0.1.0'
