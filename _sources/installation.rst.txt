Installation
============

Reference JAX package
---------------------

GraphEm supports Python 3.8 or newer. Install the current ``0.2.x`` API from
its release tag:

.. code-block:: bash

   python -m pip install "graphem-jax @ git+https://github.com/sashakolpakov/graphem.git@graphem-jax-0.2.0"

PyPI currently carries the legacy ``0.1.0`` package. A plain
``pip install graphem-jax`` therefore does not yet match the adjacency-object
examples on this site. Verify ``graphem.__version__`` when reproducing an older
environment.

For a development checkout:

.. code-block:: bash

   git clone https://github.com/sashakolpakov/graphem.git
   cd graphem
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[test,docs]"

JAX accelerator installation
----------------------------

The default PyPI installation is portable and may use the CPU. Accelerator
wheels depend on the operating system, accelerator, and driver. Follow the
`official JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
instead of guessing a CUDA wheel URL.

Verify the selected JAX backend explicitly:

.. code-block:: python

   import jax

   print(jax.default_backend())
   print(jax.devices())

The reference implementation still uses SciPy for normalized-Laplacian
initialization. For an end-to-end production CUDA path and the large-scale
reproduction suite, install
`graphem-rapids <https://github.com/sashakolpakov/graphem-rapids>`_.

Verification
------------

.. code-block:: bash

   python -c "import graphem; print(graphem.__version__)"
   python -m pytest tests -q
   python build_docs.py

Troubleshooting
---------------

``ModuleNotFoundError``
~~~~~~~~~~~~~~~~~~~~~~~

Confirm that the active interpreter and installer belong to the same virtual
environment:

.. code-block:: bash

   python -c "import sys; print(sys.executable)"
   python -m pip show graphem-jax

JAX device mismatch
~~~~~~~~~~~~~~~~~~~

Inspect ``jax.devices()`` and consult the JAX installation guide. GraphEm does
not silently convert a CPU-only JAX installation into a CUDA installation.

Out-of-memory errors
~~~~~~~~~~~~~~~~~~~~

Reduce ``sample_size``, ``batch_size``, or the graph size for the reference
package. Production-scale CUDA failures must be reported with the full graph,
configuration, backend, and memory receipt rather than silently changing the
algorithm.
