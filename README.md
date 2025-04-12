# Graphem

A graph embedding library based on JAX for efficient k-nearest neighbors and influence maximization in networks.

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

### Standard Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/igorrivin/graphem.git

# OR clone and install locally
git clone https://github.com/igorrivin/graphem.git
cd graphem
pip install .
```

### Development Installation

```bash
git clone https://github.com/igorrivin/graphem.git
cd graphem
pip install -e .  # Install in development mode
pip install -e ".[dev,profiling]"  # Install with development and profiling extras
```

### Dependencies

All required dependencies will be installed automatically. Key dependencies include:
- JAX and JAXlib for accelerated computation
- NetworkX for graph operations
- Plotly for visualization
- NumPy and SciPy for numerical operations

## Usage

### Basic Graph Embedding

```python
import numpy as np
from graphem.generators import erdos_renyi_graph
from graphem.embedder import GraphEmbedder

# Generate a random graph
edges = erdos_renyi_graph(n=200, p=0.05)
n_vertices = 200

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

## Example Scripts

The `examples/` directory contains sample scripts demonstrating different use cases:

- `basic_embedding.py`: Simple graph embedding example
- `influence_maximization.py`: Seed selection for influence maximization
- `benchmark.py`: Benchmark different graph models

## License

MIT
