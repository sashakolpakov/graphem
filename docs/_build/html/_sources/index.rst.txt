GraphEm: Graph Embedding & Influence Maximization
================================================

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/pypi/v/graphem-jax.svg
   :target: https://pypi.org/project/graphem-jax/
   :alt: PyPI

.. image:: https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/pylint.yml?branch=main&label=CI&logo=github
   :target: https://github.com/sashakolpakov/graphem/actions/workflows/pylint.yml
   :alt: CI Status

A high-performance graph embedding and influence maximization library powered by JAX, designed for scalable network analysis and research.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api_reference
   contributing

What is GraphEm?
-----------------

GraphEm is a comprehensive Python library that combines cutting-edge graph embedding techniques with influence maximization algorithms. Built on JAX for high-performance computation, it enables researchers and practitioners to analyze large-scale networks efficiently.

Key Features
~~~~~~~~~~~~

**High-Performance Computing**
   * JAX-accelerated computations with GPU/TPU support
   * JIT compilation for optimized performance
   * Memory-efficient algorithms for large networks
   * Batch processing capabilities

**Advanced Graph Embedding**
   * Spectral initialization using graph Laplacian
   * Force-directed layout refinement with customizable forces
   * 2D and 3D embedding support
   * Hierarchical position indexing for efficient k-NN search

**Influence Maximization**
   * Novel embedding-based seed selection algorithm
   * Traditional greedy algorithm for comparison
   * NDlib integration for influence spread simulation
   * Comprehensive benchmarking tools

**Graph Generation & Datasets**
   * 12+ standard graph models (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, etc.)
   * Built-in loaders for real-world datasets (SNAP, Network Repository)
   * Custom graph generators for domain-specific networks

**Visualization & Analysis**
   * Interactive 2D/3D plots with Plotly
   * Centrality correlation analysis
   * Performance benchmarking tools
   * Comprehensive reporting utilities

Quick Start Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import graphem as ge
   import networkx as nx

   # Generate a scale-free network
   edges = ge.generate_ba(n=1000, m=3)
   
   # Create and run embedding
   embedder = ge.GraphEmbedder(
       edges=edges, 
       n_vertices=1000, 
       dimension=3
   )
   embedder.run_layout(num_iterations=50)
   
   # Find influential nodes
   seeds = ge.graphem_seed_selection(embedder, k=20)
   
   # Estimate influence spread
   G = nx.Graph(edges)
   influence, _ = ge.ndlib_estimated_influence(G, seeds, p=0.1)
   print(f"Influence spread: {influence} nodes ({influence/1000:.1%})")
   
   # Visualize results
   embedder.display_layout()

Installation
~~~~~~~~~~~~

Install GraphEm using pip:

.. code-block:: bash

   pip install graphem-jax

For GPU/TPU support, see the `JAX installation guide <https://github.com/google/jax#installation>`_.

Core Components
---------------

GraphEmbedder
~~~~~~~~~~~~~

The main embedding engine that combines spectral initialization with iterative force-directed refinement:

* **Spectral Initialization**: Uses graph Laplacian eigenvectors for initial positioning
* **Force-Directed Layout**: Applies spring forces between connected nodes and repulsion between all nodes
* **Intersection Avoidance**: Prevents node overlap for cleaner visualizations
* **Adaptive Parameters**: Automatically adjusts forces based on graph structure

HPIndex
~~~~~~~

High performance index for efficient k-nearest neighbor search in high-dimensional embeddings:

* **Memory Efficient**: Memory tiling for large point sets
* **Batch Processing**: Vectorized nearest neighbor queries

Influence Maximization
~~~~~~~~~~~~~~~~~~~~~~

Advanced algorithms for identifying influential nodes in networks:

* **GraphEm Method**: Uses embedding radial distances to select diverse, influential seeds
* **Greedy Baseline**: Traditional greedy algorithm for comparison
* **Spread Simulation**: NDlib integration for accurate influence estimation

Graph Generators
~~~~~~~~~~~~~~~~

Comprehensive collection of standard and custom graph models:

.. code-block:: python

   # Classic models
   edges = ge.erdos_renyi_graph(n=500, p=0.02)
   edges = ge.generate_ba(n=500, m=3)  # Scale-free
   edges = ge.generate_ws(n=500, k=6, p=0.1)  # Small-world
   
   # Community structures
   edges = ge.generate_sbm(sizes=[100, 150, 100], p_in=0.1, p_out=0.01)
   edges = ge.generate_caveman(clique_size=10, num_cliques=5)
   
   # Specialized networks
   edges = ge.generate_geometric(n=300, radius=0.2)
   edges = ge.generate_road_network(grid_size=20, connection_prob=0.8)

Performance Characteristics
---------------------------

GraphEm is designed for high performance across different scales:

**Small Networks (< 1K nodes)**
   * Real-time embedding and visualization
   * Interactive parameter tuning
   * Comprehensive analysis possible

**Medium Networks (1K - 10K nodes)**
   * Efficient embedding with optimized parameters
   * Batch processing for multiple analyses
   * GPU acceleration recommended

**Large Networks (10K+ nodes)**
   * Memory-efficient algorithms
   * Progressive refinement strategies
   * Distributed processing capabilities

Benchmarking Results
~~~~~~~~~~~~~~~~~~~~

GraphEm shows promising performance:

* **Speed**: Much faster than the greedy algorithm for node influence maximization
* **Memory**: Memory tiling for efficiency, can process large datasets
* **Accuracy**: Strong correlation with ground-truth centrality measures


License
-------

MIT


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`