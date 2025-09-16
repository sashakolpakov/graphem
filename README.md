<p align="center">
  <img src="docs/logo.png" alt="graphem logo" height="240"/>
</p>

<h1 align="center">Graph embedding and node influence maximization</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  </a>
  <a href="https://pypi.org/project/graphem-jax/">
    <img src="https://img.shields.io/pypi/v/graphem-jax.svg" alt="PyPI"/>
  </a>
  <a href="https://pepy.tech/projects/graphem-jax">
    <img src="https://static.pepy.tech/badge/graphem-jax" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/sashakolpakov/graphem/actions/workflows/pylint.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/pylint.yml?branch=main&label=CI&logo=github" alt="CI"/>
  </a>
  <a href="https://github.com/sashakolpakov/graphem/actions/workflows/deploy_docs.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/deploy_docs.yml?branch=main&label=Docs&logo=github" alt="Docs"/>
  </a>
  <a href="https://sashakolpakov.github.io/graphem/">
    <img src="https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/graphem?label=API%20Documentation" alt="Docs Status"/>
  </a>
</p>

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

> **Note**: For GPU or TPU acceleration, JAX needs to be specifically installed with hardware support. See the [JAX documentation](https://github.com/google/jax#installation) for more details on enabling GPU/TPU support.

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
G = nx.from_edgelist(edges)
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
### Google Colab Notebook

You can check out most of the graphem features in Google Colab by running the demo notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashakolpakov/graphem/blob/main/examples/graphem_notebook.ipynb)


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
- `graph_generator_example.py` - Generate and visualize various graph embeddings
- `random_regular_example.py` - Random regular graph analysis with GraphEm
- `real_world_datasets_example.py` - Work with real world datasets (based on Facebook, arXiv, and Wikipedia data)
- `graphem_notebook.ipynb` - Interactive Jupyter notebook with examples and visualizations 

## Testing

GraphEm includes a comprehensive unit test suite that validates all core functionality using the built-in graph generators.

### Running Tests

To run the full test suite:
```bash
python -m pytest tests/
```

For verbose output:
```bash
python -m pytest tests/ -v
```

### Test Coverage

The test suite covers:

- **Graph Generators** (`test_generators.py`): All built-in graph generators including Erdős-Rényi, Barabási-Albert, Watts-Strogatz, random regular, geometric, caveman, and stochastic block models
- **Graph Embedder** (`test_embedder.py`): Core embedding functionality, layout algorithms, different dimensions, and large graph handling  
- **Influence Maximization** (`test_influence.py`): NDLib integration, seed selection, and influence estimation

### Test Requirements

Tests require the same dependencies as GraphEm plus:
- `pytest` (for running tests)
- `ndlib` (for influence maximization tests)

All tests use deterministic seeds for reproducible results.

## Benchmarking

Run comprehensive benchmarks:
```bash
python run_benchmarks.py
```

Generates performance tables and correlation analysis in Markdown and LaTeX formats.

## Documentation

Full API documentation is available [here](https://sashakolpakov.github.io/graphem/).

## Contributing

Quick start: See [CONTRIBUTING.md](CONTRIBUTING.md) for essential guidelines.

Detailed guide: [contributing documentation](https://sashakolpakov.github.io/graphem/contributing.html) for development setup, testing, and contribution guidelines.

## Citation

If you use GraphEm in research, please cite our work [![arXiv](https://img.shields.io/badge/arXiv-2506.07435-b31b1b.svg)](https://arxiv.org/abs/2506.07435)

## License

MIT
