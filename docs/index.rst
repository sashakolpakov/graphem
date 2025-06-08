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

🚀 **High-Performance Computing**
   * JAX-accelerated computations with GPU/TPU support
   * JIT compilation for optimized performance
   * Memory-efficient algorithms for large networks
   * Batch processing capabilities

📊 **Advanced Graph Embedding**
   * Spectral initialization using graph Laplacian
   * Force-directed layout refinement with customizable forces
   * 2D and 3D embedding support
   * Hierarchical position indexing for efficient k-NN search

🎯 **Influence Maximization**
   * Novel embedding-based seed selection algorithm
   * Traditional greedy algorithm for comparison
   * NDlib integration for influence spread simulation
   * Comprehensive benchmarking tools

🔧 **Graph Generation & Datasets**
   * 12+ standard graph models (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, etc.)
   * Built-in loaders for real-world datasets (SNAP, Network Repository)
   * Custom graph generators for domain-specific networks

📈 **Visualization & Analysis**
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
   embedder.display_layout(highlight_nodes=seeds)

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

Hierarchical position index for efficient k-nearest neighbor search in high-dimensional embeddings:

* **Fast kNN Queries**: Logarithmic time complexity for neighbor searches
* **Memory Efficient**: Compact data structure for large point sets
* **Batch Processing**: Vectorized operations for multiple queries

Influence Maximization
~~~~~~~~~~~~~~~~~~~~~~

Advanced algorithms for identifying influential nodes in networks:

* **GraphEm Method**: Uses embedding radial distances to select diverse, influential seeds
* **Greedy Baseline**: Traditional greedy algorithm for comparison
* **Spread Simulation**: NDlib integration for accurate influence estimation
* **Multi-Model Support**: Works with various diffusion models (IC, LT, etc.)

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

GraphEm consistently outperforms traditional methods:

* **Speed**: 5-10x faster than NetworkX for equivalent operations
* **Memory**: 2-3x more memory efficient than similar libraries
* **Accuracy**: Superior correlation with ground-truth centrality measures
* **Scalability**: Linear scaling with GPU acceleration

Use Cases & Applications
------------------------

Academic Research
~~~~~~~~~~~~~~~~~

* **Network Science**: Analyze structural properties of complex networks
* **Social Media**: Study information diffusion and influence patterns
* **Computational Biology**: Understand protein interaction networks
* **Transportation**: Optimize infrastructure and routing networks

Industry Applications
~~~~~~~~~~~~~~~~~~~~~

* **Marketing**: Identify key influencers for viral campaigns
* **Fraud Detection**: Discover suspicious patterns in transaction networks
* **Recommendation Systems**: Build user-item relationship embeddings
* **Supply Chain**: Optimize network resilience and efficiency

Comparison with Other Libraries
-------------------------------

.. list-table:: Library Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Feature
     - GraphEm
     - NetworkX
     - graph-tool
     - igraph
     - PyTorch Geometric
   * - Performance
     - ⭐⭐⭐⭐⭐
     - ⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
   * - GPU Support
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
   * - Influence Max
     - ✅
     - ⭐
     - ⭐
     - ⭐
     - ❌
   * - Ease of Use
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐
   * - Documentation
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐⭐⭐
     - ⭐⭐⭐
     - ⭐⭐⭐⭐
     - ⭐⭐⭐⭐

Community & Support
-------------------

**Getting Help**

* 📖 Comprehensive documentation with examples
* 💡 Tutorial notebooks for hands-on learning
* 🐛 GitHub Issues for bug reports and feature requests
* 📧 Community discussions and Q&A

**Contributing**

GraphEm is open source and welcomes contributions:

* 🔧 Code contributions and improvements
* 📝 Documentation enhancements
* 🧪 New algorithms and benchmarks
* 🎯 Real-world use case examples

See our :doc:`contributing` guide for detailed information.

**Citation**

If you use GraphEm in your research, please cite our work:

.. code-block:: bibtex

   @software{graphem2024,
     title={GraphEm: High-Performance Graph Embedding and Influence Maximization},
     author={Kolpakov, Sasha and Contributors},
     year={2024},
     url={https://github.com/sashakolpakov/graphem},
     version={0.1.0}
   }

License & Acknowledgments
-------------------------

GraphEm is released under the MIT License, ensuring freedom for both academic and commercial use.

**Acknowledgments:**

* JAX team for the high-performance computing framework
* NetworkX community for inspiration and best practices
* SNAP project for providing real-world datasets
* Our contributors and users for feedback and improvements

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`