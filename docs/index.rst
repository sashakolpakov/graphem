GraphEm: geometric graph embedding and radial node ranking
==========================================================

.. image:: logo.png
   :alt: GraphEm logo
   :align: center
   :width: 320px

GraphEm builds a low-dimensional graph layout whose radial coordinate can be
used as an empirical node-ranking score. This site documents the portable JAX
reference package, distributed as ``graphem-jax``.

The H100-qualified production CUDA implementation is maintained separately in
`graphem-rapids <https://github.com/sashakolpakov/graphem-rapids>`_. It uses the
same GraphEm algorithmic family but has a different execution and evidence
contract.

.. important::

   Radial rank is a heuristic proxy. It is not guaranteed to reproduce every
   centrality measure and it is not a guaranteed influence-maximization method.
   Validate it against task-appropriate baselines on held-out graphs or seeds.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   tutorials
   api_reference
   contributing

Project links
-------------

* `Source repository <https://github.com/sashakolpakov/graphem>`_
* `CUDA implementation <https://github.com/sashakolpakov/graphem-rapids>`_
* `Issue tracker <https://github.com/sashakolpakov/graphem/issues>`_
* `Manuscript <https://arxiv.org/abs/2506.07435>`_
* `PyPI package <https://pypi.org/project/graphem-jax/>`_

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
