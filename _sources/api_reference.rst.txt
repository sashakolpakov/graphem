API reference
=============

Core embedder
-------------

.. autoclass:: graphem.embedder.GraphEmbedder
   :members:
   :show-inheritance:

Nearest-neighbor index
----------------------

.. autoclass:: graphem.index.HPIndex
   :members:
   :show-inheritance:

Graph generators
----------------

Public generators return SciPy CSR sparse adjacency objects. When
``generate_sbm(..., labels=True)`` is requested, it returns the adjacency object
and the block-label array as a tuple.

.. automodule:: graphem.generators
   :members:
   :undoc-members:
   :show-inheritance:

Radial selection and influence evaluation
-----------------------------------------

These functions are evaluation conveniences. They do not establish that radial
selection is optimal for influence maximization.

.. automodule:: graphem.influence
   :members:
   :undoc-members:
   :show-inheritance:

Datasets
--------

.. automodule:: graphem.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Visualization and statistics
----------------------------

.. automodule:: graphem.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark helpers
-----------------

.. automodule:: graphem.benchmark
   :members:
   :undoc-members:
   :show-inheritance:
