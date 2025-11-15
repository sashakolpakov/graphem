# Changelog

All notable changes to GraphEm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-15

### Added
- **New graph generators** (`graphem/generators.py`):
  - `generate_delaunay_triangulation()`: Generate planar graphs with triangular faces based on Delaunay triangulation
  - `generate_complete_bipartite_graph()`: Generate complete bipartite graphs

### Changed
- **BREAKING: API aligned with CUDA version** - Major API changes for consistency:
  - **All generators now return sparse adjacency matrices** (scipy.sparse.csr_matrix) instead of edge lists
  - Renamed `erdos_renyi_graph()` → `generate_er()` for consistency with other generators
  - `GraphEmbedder` now accepts `adjacency` (sparse matrix) instead of `edges` + `n_vertices`
  - `GraphEmbedder` parameter renamed: `my_logger` → `logger_instance`
  - `generate_bipartite_graph()` now accepts `p` and `seed` parameters for better control
  - `compute_vertex_degrees()` now accepts adjacency matrix instead of edge list

- **GraphEmbedder improvements**:
  - Added `seed` parameter for reproducibility
  - Added `get_positions()` method that returns numpy array
  - Made `positions` a property (internally uses `_positions`)
  - Automatically infers number of vertices from adjacency matrix shape
  - Improved adjacency matrix validation

### Fixed
- **Critical bug fix in influence maximization** (`graphem/influence.py`): Fixed `ndlib_estimated_influence()` function to correctly initialize seed nodes using NDlib's proper API. Previously, the function was using an incorrect configuration method that resulted in seeds not being properly set, leading to inaccurate influence estimations. The fix ensures:
  - Seeds are now correctly initialized using `config.add_model_initial_configuration("Infected", seeds)` instead of the incorrect `config.add_node_configuration("status", seed, 1)`
  - Influenced node counts are now correctly retrieved from `iterations[-1]['node_count'].get(2, 0)` instead of manually iterating through status values
  - All influence maximization benchmarks and comparisons now produce accurate results
  - High-degree seeds now correctly show higher influence than low-degree seeds

This bug affected all influence maximization functionality including `graphem_seed_selection()`, `greedy_seed_selection()`, and benchmark comparisons. Users should re-run any influence maximization experiments performed with previous versions.

### Migration Guide
To upgrade from 0.1.x to 0.2.0:

**Generators:**
```python
# OLD (0.1.x)
edges = ge.erdos_renyi_graph(n=100, p=0.1)  # Returns edge list
n = 100

# NEW (0.2.0)
adj = ge.generate_er(n=100, p=0.1)  # Returns sparse adjacency matrix
n = adj.shape[0]  # Infer from matrix
```

**GraphEmbedder:**
```python
# OLD (0.1.x)
embedder = ge.GraphEmbedder(edges=edges, n_vertices=n, n_components=2, my_logger=logger)

# NEW (0.2.0)
embedder = ge.GraphEmbedder(adjacency=adj, n_components=2, logger_instance=logger)
```

**compute_vertex_degrees:**
```python
# OLD (0.1.x)
degrees = ge.compute_vertex_degrees(n, edges)

# NEW (0.2.0)
degrees = ge.compute_vertex_degrees(adj)
```

## [Previous Releases]

For release history before this changelog was established, see the [GitHub Releases](https://github.com/igorrivin/graphem/releases) page.
