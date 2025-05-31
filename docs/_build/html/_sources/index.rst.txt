Welcome to GraphEm's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

GraphEm is a graph embedding library based on JAX for efficient centrality measures approximation and influence maximization in networks.

Overview
--------

Graphem is a Python library for graph visualization and analysis, with a focus on efficient embedding of large networks. It uses JAX for accelerated computation and provides tools for influence maximization in networks.

Key features:

* Fast graph embedding using Laplacian embedding
* Efficient k-nearest neighbors search with JAX
* Various graph generation models
* Tools for influence maximization
* Graph visualization with Plotly
* Benchmarking tools for comparing graph metrics

Installation
------------

To install from PyPI::

    pip install graphem-jax

To install from the GitHub repository::

    pip install git+https://github.com/sashakolpakov/graphem.git

Quick Start
-----------

Basic Graph Embedding
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Influence Maximization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

API Reference
=============

.. automodule:: graphem
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`