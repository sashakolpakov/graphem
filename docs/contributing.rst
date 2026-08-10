Contributing to GraphEm
=======================

Set up a development environment
--------------------------------

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/graphem.git
   cd graphem
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[test,docs]"

Run the core Python and Sphinx gates
------------------------------------

.. code-block:: bash

   python -m pytest tests -v
   pylint --output-format=colorized --rcfile=.github/workflows/.pylintrc $(git ls-files '*.py' ':!:setup.py')
   python -c "import graphem, graphem.benchmark, graphem.datasets, graphem.generators, graphem.influence, graphem.visualization"
   python build_docs.py

CI imports the public modules with the real installed dependencies before
``build_docs.py`` runs Sphinx in nitpicky mode with warnings treated as errors.
CI also checks Markdown style and external links.

Documentation rules
-------------------

* Update every affected Markdown and Sphinx page in the same change.
* Keep examples executable against the current public API.
* Distinguish the JAX reference package from the production CUDA package.
* Report empirical limitations; do not turn a benchmark observation into a
  universal performance, centrality, or influence claim.
* Link to ``sashakolpakov`` repository and Pages URLs, not historical mirrors.

Benchmark evidence
------------------

Each reproduction cell is independently replaceable. A replacement must retain
the parent manifest, reason, old artifact, new artifact, source and environment
receipts, and an independent cell audit. Never overwrite an accepted result or
silently recompute unrelated cells.

Submitting changes
------------------

#. Create a focused branch.
#. Add tests and documentation.
#. Run the required checks.
#. Open a pull request describing behavior, evidence, and limitations.
#. Wait for required Actions checks and review before merging.

Report bugs in the `issue tracker
<https://github.com/sashakolpakov/graphem/issues>`_.
