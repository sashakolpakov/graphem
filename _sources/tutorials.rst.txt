Tutorials
=========

Graph families
--------------

The generator module returns sparse adjacency matrices with stable integer node
labels.

.. code-block:: python

   import graphem as ge

   graphs = {
       "ER": ge.generate_er(n=300, p=0.02, seed=1),
       "small world": ge.generate_ws(n=300, k=6, p=0.2, seed=1),
       "preferential attachment": ge.generate_ba(n=300, m=3, seed=1),
       "grid": ge.generate_road_network(width=20, height=15),
       "tree": ge.generate_balanced_tree(r=2, h=7),
   }

   for name, adjacency in graphs.items():
       print(name, adjacency.shape, adjacency.nnz // 2)

Radial ranking and ties
-----------------------

Rank by decreasing radius and use the node ID as a stable tie breaker:

.. code-block:: python

   import numpy as np

   adjacency = ge.generate_power_cluster(n=500, m=3, p=0.4, seed=9)
   embedder = ge.GraphEmbedder(adjacency, n_components=3, seed=9, verbose=False)
   positions = embedder.run_layout(num_iterations=40)
   radii = np.linalg.norm(positions, axis=1)
   order = np.lexsort((np.arange(len(radii)), -radii))
   print(order[:20])

Different graph families need not have the same relationship between radial
rank and centrality. Grids and highly symmetric graphs are especially useful
negative or stress-test cases.

Centrality evaluation
---------------------

.. code-block:: python

   import networkx as nx
   from scipy.stats import spearmanr

   graph = nx.from_scipy_sparse_array(adjacency)
   degree = np.asarray([graph.degree(node) for node in graph.nodes()])
   rho = spearmanr(radii, degree).statistic
   print(f"Spearman rho(radius, degree) = {rho:.4f}")

For a complete evaluation, report the exact graph identity, parameters, raw
position and score hashes, all requested centralities, and top-k overlaps. A
single favorable target is not evidence of a universal centrality proxy.

Influence experiments
---------------------

The radial selector is a heuristic. Compare it to a greedy or CELF-style
baseline on identical propagation worlds:

.. code-block:: python

   import networkx as nx

   adjacency = ge.generate_er(n=128, p=0.05, seed=5)
   graph = nx.from_scipy_sparse_array(adjacency)

   radial_embedder = ge.GraphEmbedder(adjacency, seed=5, verbose=False)
   radial = ge.graphem_seed_selection(radial_embedder, k=10, num_iterations=20)
   greedy, evaluations = ge.greedy_seed_selection(
       graph,
       k=10,
       p=0.1,
       iterations_count=200,
   )
   print(radial, greedy, evaluations)

The convenience functions in this reference package do not provide paired
propagation worlds. Use the reproduction harness for paired scientific claims.

Real-world datasets
-------------------

.. code-block:: python

   import graphem as ge
   from graphem.datasets import list_available_datasets

   print(list_available_datasets())
   vertices, edges = ge.load_dataset("facebook_combined")
   print(vertices.shape, edges.shape)

Dataset downloads are external inputs. Record the source URL, compressed and
decompressed hashes, parsing rules, component selection, and dense relabeling in
reproducible work.

Replacing one benchmark cell
----------------------------

Large reproduction matrices use one content-addressed result and audit shard per
cell. A corrected or extended cell publishes a successor manifest containing:

#. the parent manifest;
#. the replacement reason or issue;
#. source, image, environment, and input receipts;
#. the old and new artifacts; and
#. an independent audit.

Unselected cells remain byte-identical, and tables are rendered from the chosen
manifest rather than edited by hand.
