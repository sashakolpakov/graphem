API Reference
=============

This section contains detailed documentation for all GraphEm modules and functions.

Core Classes
------------

GraphEmbedder
~~~~~~~~~~~~~

**graphem.embedder.GraphEmbedder** - JAX-based graph embedding using Laplacian embedding with spring forces and intersection avoidance.

Main class for creating graph embeddings. Combines Laplacian eigenvectors with force-directed layout optimization to create visually appealing and structurally meaningful embeddings.

Features automatic parameter safeguards: ``sample_size`` is automatically limited to the number of edges, and ``batch_size`` is automatically limited to the number of vertices, eliminating the need for manual size calculations.

Key methods:
- ``__init__(edges, n_vertices, n_components=2, ...)`` - Initialize embedder with graph structure
- ``run_layout(num_iterations=100)`` - Execute layout algorithm for specified iterations
- ``display_layout(edge_width=1, node_size=3, ...)`` - Visualize embedding using Plotly

.. autoclass:: graphem.embedder.GraphEmbedder
   :members:
   :undoc-members:
   :show-inheritance:

HPIndex
~~~~~~~

**graphem.index.HPIndex** - High-performance k-nearest neighbors search with JAX acceleration.

Efficient batched k-NN implementation for large datasets with memory optimization through tiling.

Key methods:
- ``knn_tiled(x, y, k=5, ...)`` - Find k nearest neighbors with batched processing

.. autoclass:: graphem.index.HPIndex
   :members:
   :undoc-members:
   :show-inheritance:

Graph Generators
----------------

**graphem.generators** - Generate various graph types for testing and experimentation.

Provides NetworkX-based generators for standard graph models including random graphs, scale-free networks, small-world graphs, and more.

Key functions:
- ``erdos_renyi_graph(n, p)`` - Random graph with edge probability p
- ``generate_sbm(n_per_block, num_blocks, p_in, p_out)`` - Stochastic block model
- ``generate_ba(n, m)`` - Barabási-Albert preferential attachment
- ``generate_ws(n, k, p)`` - Watts-Strogatz small-world
- ``generate_scale_free(n, ...)`` - Scale-free network
- ``generate_geometric(n, radius)`` - Random geometric graph

.. automodule:: graphem.generators
   :members:
   :undoc-members:
   :show-inheritance:

Influence Maximization
----------------------

**graphem.influence** - Seed selection algorithms for influence maximization in networks.

Implements GraphEm-based seed selection using radial distances from embedding origin, plus traditional greedy methods with NDlib simulation.

Key functions:
- ``graphem_seed_selection(embedder, k)`` - Select seeds based on radial distances
- ``greedy_seed_selection(G, k, p)`` - Traditional greedy algorithm
- ``ndlib_estimated_influence(G, seeds, p)`` - Evaluate influence using Independent Cascades

.. automodule:: graphem.influence
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Utilities
-----------------

**graphem.datasets** - Load real-world network datasets from various sources.

Download and process datasets from SNAP, Network Repository, and Semantic Scholar. Handles automatic downloading, extraction, and format conversion.

Key classes:
- ``SNAPDataset(dataset_name)`` - SNAP collection (Facebook, Wikipedia, arXiv, etc.)
- ``NetworkRepositoryDataset(dataset_name)`` - Network Repository collection
- ``SemanticScholarDataset(dataset_name)`` - Citation networks

Key functions:
- ``load_dataset(dataset_name)`` - Load any dataset by name
- ``load_dataset_as_networkx(dataset_name)`` - Load as NetworkX graph
- ``list_available_datasets()`` - Show all available datasets

.. automodule:: graphem.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

**graphem.visualization** - Statistical analysis and plotting utilities for embeddings.

Calculate correlations between embedding radial distances and network centrality measures, with bootstrap confidence intervals.

Key functions:
- ``report_corr(name, radii, centrality)`` - Correlation with confidence intervals
- ``report_full_correlation_matrix(radii, ...)`` - Multiple centrality correlations
- ``plot_radial_vs_centrality(radii, centralities, names)`` - Scatter plots with trendlines
- ``display_benchmark_results(results)`` - Format benchmark output

.. automodule:: graphem.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Benchmarking
------------

**graphem.benchmark** - Performance evaluation and comparative analysis tools.

Comprehensive benchmarking for embedding quality, centrality correlations, and influence maximization effectiveness.

Key functions:
- ``run_benchmark(graph_generator, params)`` - Basic embedding benchmark
- ``benchmark_correlations(graph_generator, params)`` - Centrality correlation analysis
- ``run_influence_benchmark(graph_generator, params)`` - Compare influence methods

.. automodule:: graphem.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

Complete Module Reference
-------------------------

graphem.embedder module
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.embedder
   :members:
   :undoc-members:
   :show-inheritance:

graphem.generators module
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.generators
   :members:
   :undoc-members:
   :show-inheritance:

graphem.influence module
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.influence
   :members:
   :undoc-members:
   :show-inheritance:

graphem.index module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.index
   :members:
   :undoc-members:
   :show-inheritance:

graphem.datasets module
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.datasets
   :members:
   :undoc-members:
   :show-inheritance:

graphem.visualization module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.visualization
   :members:
   :undoc-members:
   :show-inheritance:

graphem.benchmark module
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphem.benchmark
   :members:
   :undoc-members:
   :show-inheritance: