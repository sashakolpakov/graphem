Quick Start Guide
=================

This guide will get you up and running with GraphEm in just a few minutes.

Installation
------------

Install GraphEm using pip:

.. code-block:: bash

    pip install graphem-jax

For GPU/TPU acceleration (optional but recommended for large graphs), see the `JAX installation guide <https://github.com/google/jax#installation>`_.

Your First Graph Embedding
---------------------------

Let's start with a simple example of embedding a random graph:

.. code-block:: python

    import graphem as ge
    import numpy as np

    # Generate a random graph
    edges = ge.erdos_renyi_graph(n=200, p=0.05)
    
    # Create an embedder
    embedder = ge.GraphEmbedder(
        edges=edges,
        n_vertices=200,
        n_components=3,   # 3D embedding
        L_min=10.0,       # Minimum edge length
        k_attr=0.5,       # Attraction force
        k_inter=0.1,      # Repulsion force
        n_neighbors=15    # Nearest neighbors
    )
    
    # Compute the embedding
    embedder.run_layout(num_iterations=40)
    
    # Visualize the result
    embedder.display_layout(edge_width=0.5, node_size=5)

Understanding the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **n_components**: Embedding space dimension (2D or 3D)
* **L_min**: Controls minimum distance between connected nodes
* **k_attr**: Strength of attractive forces between connected nodes
* **k_inter**: Strength of repulsive forces between all nodes
* **n_neighbors**: Number of nearest neighbors for efficient force computation

Graph Generation
----------------

GraphEm provides various graph generators:

.. code-block:: python

    # Scale-free network (Barabási–Albert)
    edges = ge.generate_ba(n=500, m=3)
    
    # Small-world network (Watts–Strogatz)
    edges = ge.generate_ws(n=500, k=6, p=0.1)
    
    # Stochastic block model
    edges = ge.generate_sbm(n_per_block=100, num_blocks=3, p_in=0.1, p_out=0.01)
    
    # Random regular graph
    edges = ge.generate_random_regular(n=300, d=4)

Complete Graph Generator Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GraphEm provides 12+ graph generators for different network types:

.. code-block:: python

    # Random graphs
    edges = ge.erdos_renyi_graph(n=500, p=0.02)  # Random graph
    edges = ge.generate_random_regular(n=300, d=4)  # Regular degree
    edges = ge.generate_geometric(n=200, radius=0.2)  # Geometric graph

    # Scale-free and complex networks
    edges = ge.generate_ba(n=500, m=3)  # Barabási-Albert
    edges = ge.generate_scale_free(n=400, alpha=0.41, beta=0.54)  # Scale-free
    edges = ge.generate_power_cluster(n=500, m=3, p=0.5)  # Powerlaw cluster

    # Small-world networks
    edges = ge.generate_ws(n=500, k=6, p=0.1)  # Watts-Strogatz

    # Community structures
    edges = ge.generate_sbm(n_per_block=100, num_blocks=3, p_in=0.1, p_out=0.01)
    edges = ge.generate_caveman(l=10, k=10)  # Connected caveman
    edges = ge.generate_relaxed_caveman(l=10, k=10, p=0.1)  # Relaxed caveman

    # Specialized networks
    edges = ge.generate_bipartite_graph(n_top=100, n_bottom=150)  # Bipartite
    edges = ge.generate_balanced_tree(r=3, h=8)  # Balanced tree
    edges = ge.generate_road_network(width=20, height=20)  # Grid-like road network

Working with Real Data
----------------------

Load and analyze real-world networks:

.. code-block:: python

    # Load a dataset (includes several network datasets)
    vertices, edges = ge.load_dataset('snap-ca-GrQc')  # Collaboration network
    n_vertices = len(vertices)
    
    # Create embedder for larger networks
    embedder = ge.GraphEmbedder(
        edges=edges,
        n_vertices=n_vertices,
        n_components=2,
        n_neighbors=20,     # More neighbors for denser graphs
        sample_size=512,    # Larger sample for accuracy
        batch_size=2048     # Larger batches for efficiency
    )
    
    embedder.run_layout(num_iterations=100)
    embedder.display_layout()

Influence Maximization
-----------------------

Identify influential nodes:

.. code-block:: python

    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    G.add_edges_from(edges)

    # Fast: embedding-based selection
    seeds_graphem = ge.graphem_seed_selection(embedder, k=10, num_iterations=20)

    # Accurate: greedy algorithm
    seeds_greedy, total_iters = ge.greedy_seed_selection(G, k=10, p=0.1, iterations_count=100)

    # Evaluate influence spread (Independent Cascades model)
    influence, iters = ge.ndlib_estimated_influence(G, seeds_graphem, p=0.1, iterations_count=200)

    print(f"Influenced: {influence}/{n_vertices} nodes ({influence/n_vertices:.1%})")

Benchmarking and Analysis
-------------------------

Compare different centrality measures:

.. code-block:: python

    from graphem.benchmark import benchmark_correlations
    from graphem.visualization import report_full_correlation_matrix
    
    # Run comprehensive benchmark
    results = benchmark_correlations(
        graph_generator=ge.generate_ba,
        graph_params={'n': 300, 'm': 3},
        n_components=3,
        num_iterations=50
    )
    
    # Display correlation matrix
    correlation_matrix = report_full_correlation_matrix(
        results['radii'],           # Embedding-based centrality
        results['degree'],          # Degree centrality
        results['betweenness'],     # Betweenness centrality
        results['eigenvector'],     # Eigenvector centrality
        results['pagerank'],        # PageRank
        results['closeness'],       # Closeness centrality
        results['node_load']        # Load centrality
    )

Performance Tips
----------------

**For Large Graphs (>10k nodes):**

.. code-block:: python

    embedder = ge.GraphEmbedder(
        edges=edges,
        n_vertices=n_vertices,
        n_components=2,       # 2D is faster than 3D
        n_neighbors=10,       # Fewer neighbors = faster
        sample_size=256,      # Automatically limited to len(edges)
        batch_size=4096,      # Automatically limited to n_vertices
        verbose=False         # Disable progress bars
    )

**GPU Acceleration:**

GraphEm automatically uses GPU if JAX detects CUDA:

.. code-block:: python

    import jax
    print("Available devices:", jax.devices())  # Check for GPU
    
    # Force CPU usage if needed
    with jax.default_device(jax.devices('cpu')[0]):
        embedder.run_layout(num_iterations=50)

**Memory Management:**

For very large graphs, process in chunks:

.. code-block:: python

    # For graphs with >100k nodes, consider reducing parameters
    embedder = ge.GraphEmbedder(
        edges=edges,
        n_vertices=n_vertices,
        n_neighbors=5,        # Minimum viable k
        sample_size=128,      # Automatically limited to len(edges)
        batch_size=1024       # Automatically limited to n_vertices
    )

Next Steps
----------

* Explore the :doc:`tutorials` for detailed examples
* Check the :doc:`api_reference` for complete documentation
* See :doc:`examples` for real-world use cases
* Read :doc:`contributing` to help improve GraphEm