Quick start
===========

Create an embedding
-------------------

Built-in generators return SciPy CSR sparse adjacency objects.

.. code-block:: python

   import numpy as np
   import graphem as ge

   adjacency = ge.generate_er(n=500, p=0.01, seed=7)
   embedder = ge.GraphEmbedder(
       adjacency=adjacency,
       n_components=3,
       L_min=1.0,
       k_attr=0.2,
       k_inter=0.5,
       n_neighbors=10,
       sample_size=256,
       batch_size=500,
       seed=7,
       verbose=False,
   )
   positions = embedder.run_layout(num_iterations=50)
   radii = np.linalg.norm(positions, axis=1)
   top_nodes = np.lexsort((np.arange(len(radii)), -radii))[:10]
   print(top_nodes)

``positions`` has shape ``(n_vertices, n_components)``. Use
``embedder.get_positions()`` to retrieve the current layout without performing
additional iterations.

Visualize a two-dimensional layout
----------------------------------

.. code-block:: python

   adjacency = ge.generate_ws(n=300, k=6, p=0.2, seed=4)
   embedder = ge.GraphEmbedder(adjacency, n_components=2, seed=4)
   embedder.run_layout(num_iterations=30)
   embedder.display_layout(node_size=4)

Evaluate radial seed selection
------------------------------

.. code-block:: python

   import networkx as nx

   adjacency = ge.generate_er(n=128, p=0.05, seed=3)
   graph = nx.from_scipy_sparse_array(adjacency)
   embedder = ge.GraphEmbedder(adjacency, seed=3, verbose=False)

   radial_seeds = ge.graphem_seed_selection(
       embedder,
       k=10,
       num_iterations=20,
   )
   spread, steps = ge.ndlib_estimated_influence(
       graph,
       radial_seeds,
       p=0.1,
       iterations_count=200,
   )
   print(spread, steps)

This evaluates one stochastic Independent-Cascade trajectory. A scientific
comparison must use shared propagation worlds or enough independent samples and
must report uncertainty.

Measure centrality correlation
------------------------------

.. code-block:: python

   from graphem.benchmark import benchmark_correlations

   result = benchmark_correlations(
       ge.generate_er,
       graph_params={"n": 200, "p": 0.05, "seed": 11},
       n_components=3,
       num_iterations=40,
   )
   print(result)

Next steps
----------

* :doc:`tutorials` explains graph families, ranking, and honest evaluation.
* :doc:`api_reference` lists public classes and functions.
* :doc:`installation` distinguishes the reference JAX and production CUDA paths.
