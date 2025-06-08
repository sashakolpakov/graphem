# GraphEm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/graphem-jax.svg)](https://pypi.org/project/graphem-jax/)
[![CI](https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/pylint.yml?branch=main&label=CI&logo=github)](https://github.com/sashakolpakov/graphem/actions/workflows/pylint.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/deploy_docs.yml?branch=main&label=Docs&logo=github)](https://github.com/sashakolpakov/graphem/actions/workflows/deploy_docs.yml)
[![API Documentation](https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/graphem?label=API%20Documentation)](https://sashakolpakov.github.io/graphem/)

Graph embedding and influence maximization library using JAX.

## Features

- **Graph Embedding**: Laplacian-based layout with force-directed refinement
- **JAX Backend**: GPU/TPU acceleration for large graphs
- **Influence Maximization**: Novel embedding-based seed selection algorithm
- **Graph Generators**: Standard models (Erdős–Rényi, Barabási–Albert, Watts-Strogatz, etc.)
- **Visualization**: Interactive 2D/3D plots with Plotly
- **Benchmarking**: Centrality correlation analysis and performance testing
- **Datasets**: Built-in loaders for SNAP and Network Repository datasets

## Installation

```bash
pip install graphem-jax
```

For GPU/TPU acceleration, see the [JAX installation guide](https://github.com/google/jax#installation).

From source:
```bash
pip install git+https://github.com/sashakolpakov/graphem.git
```

## Quick Start

### Graph Embedding

```python
import graphem as ge

# Generate graph
edges = ge.erdos_renyi_graph(n=500, p=0.01)

# Create embedder
embedder = ge.GraphEmbedder(
    edges=edges,
    n_vertices=500,
    dimension=3
)

# Compute layout
embedder.run_layout(num_iterations=50)

# Visualize
embedder.display_layout()
```

### Influence Maximization

```python
# Select influential nodes
seeds = ge.graphem_seed_selection(embedder, k=10)

# Estimate influence spread
import networkx as nx
G = nx.Graph(edges)
influence, _ = ge.ndlib_estimated_influence(G, seeds, p=0.1)
print(f"Influence: {influence} nodes ({influence/500:.1%})")
```

### Benchmarking

```python
from graphem.benchmark import benchmark_correlations

# Compare embedding radii with centrality measures
results = benchmark_correlations(
    ge.erdos_renyi_graph,
    graph_params={'n': 200, 'p': 0.05},
    dim=3,
    num_iterations=40
)

# Display correlation matrix
ge.report_full_correlation_matrix(
    results['radii'],
    results['degree'],
    results['betweenness'],
    results['eigenvector'],
    results['pagerank'],
    results['closeness'],
    results['node_load']
)
```

## Key Components

### Core Class

- **`GraphEmbedder`**: Main embedding engine with Laplacian initialization and force-directed layout

### Algorithms

- **Graph embedding**: Spectral initialization + spring forces + intersection avoidance
- **Influence maximization**: Radial distance-based seed selection vs traditional greedy
- **Generators**: 12+ graph models including SBM, small-world, scale-free

### Datasets

Built-in access to standard network datasets:
- Stanford Network Analysis Project
- Network Repository

## Examples

The `examples/` directory contains:
- `graph_generator_test.py` - Test all graph generators
- `random_regular_test.py` - Random regular graph analysis
- `real_world_datasets_test.py` - Work with real datasets
- `graphem_notebook.ipynb` - Interactive Jupyter notebook

## Benchmarking

Run comprehensive benchmarks:
```bash
python run_benchmarks.py
```

Generates performance tables and correlation analysis in Markdown and LaTeX formats.

## Documentation

Full API documentation: **[https://sashakolpakov.github.io/graphem/](https://sashakolpakov.github.io/graphem/)**

## Contributing

See [docs/contributing.rst](docs/contributing.rst) for development setup, testing, and contribution guidelines.

## Citation

If you use GraphEm in research, please cite our work (paper forthcoming).

## License

MIT License - see [LICENSE](LICENSE) for details.