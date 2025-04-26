# GraphEm

A graph embedding library based on JAX for efficient centrality measures approximation and influence maximization in networks.

## Overview

Graphem is a Python library for graph visualization and analysis, with a focus on efficient embedding of large networks. It uses JAX for accelerated computation and provides tools for influence maximization in networks.

Key features:
- Fast graph embedding using Laplacian embedding
- Efficient k-nearest neighbors search with JAX
- Various graph generation models
- Tools for influence maximization
- Graph visualization with Plotly
- Benchmarking tools for comparing graph metrics

## Installation

At the moment, you can install from the repository only. 

```bash
pip install git+https://github.com/sashakolpakov/graphem.git
```

Soon a python package will be made available. 

## Usage

### Basic Graph Embedding

```python
from graphem.generators import erdos_renyi_graph
from graphem.embedder import GraphEmbedder

# Generate a random graph
n_vertices = 200
edges = erdos_renyi_graph(n=n_vertices, p=0.05)


# Create an embedder
embedder = GraphEmbedder(
    edges=edges,
    n_vertices=n_vertices,
    dimension=3,  # 3D embedding
    L_min=10.0,   # Minimum edge length
    k_attr=0.5,   # Attraction force constant
    k_inter=0.1,  # Repulsion force constant
    knn_k=15      # Number of nearest neighbors
)

# Run the layout algorithm
embedder.run_layout(num_iterations=40)

# Visualize the graph
embedder.display_layout(edge_width=0.5, node_size=5)
```

### Influence Maximization

```python
import networkx as nx
from graphem.influence import graphem_seed_selection, ndlib_estimated_influence

# This is supposed to be added to the previous code (like the above example) ...

# Convert edges to NetworkX graph
G = nx.Graph()
G.add_nodes_from(range(n_vertices))
G.add_edges_from(edges)

# Select seed nodes using the Graphem method
seeds = graphem_seed_selection(embedder, k=10, num_iterations=20)

# Estimate influence
influence, iterations = ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=200)
print(f"Estimated influence: {influence} nodes ({influence/n_vertices:.2%} of the graph)")
```

### Benchmarking

```python
from graphem.generators import erdos_renyi_graph
from graphem.benchmark import benchmark_correlations
from graphem.visualization import report_full_correlation_matrix

# Run benchmark to calculate correlations
results = benchmark_correlations(
    erdos_renyi_graph,
    {'n': 200, 'p': 0.05},
    dim=3,
    num_iterations=40
)

# Display correlation matrix
corr_matrix = report_full_correlation_matrix(
    results['radii'],
    results['degree'],
    results['betweenness'],
    results['eigenvector'],
    results['pagerank'],
    results['closeness'],
    results['edge_betweenness']
)
```

## Benchmarking Script

The root directory contain `run_benchmarks.py` that runs all available tests and benchmarks in the library and
generates nicely formatted result tables in Markdown and LaTeX.

## Example Scripts

The `examples/` directory contains sample scripts demonstrating different use cases:

- `graph_generator_test.py`: Test script for various graph generators
- `random_regular_test.py`: Focused test script for random regular graphs
- `real_world_datasets_test.py`: Test script for working with real-world datasets

## License

MIT
