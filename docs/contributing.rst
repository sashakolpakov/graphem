Contributing to GraphEm
======================

We welcome contributions to GraphEm! This guide will help you get started with development, testing, and submitting high-quality contributions.

Development Environment Setup
-----------------------------

Prerequisites
~~~~~~~~~~~~~

Before contributing, ensure you have:

* Python 3.8+ (Python 3.9+ recommended)
* Git
* A virtual environment manager (venv, conda, or similar)

Initial Setup
~~~~~~~~~~~~~

1. **Fork and clone the repository:**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/graphem.git
   cd graphem
   git remote add upstream https://github.com/sashakolpakov/graphem.git

2. **Create and activate a virtual environment:**

.. code-block:: bash

   python -m venv graphem-dev
   source graphem-dev/bin/activate  # On Windows: graphem-dev\Scripts\activate

3. **Install in development mode:**

.. code-block:: bash

   pip install --upgrade pip
   pip install -e ".[dev,docs]"  # Install with development dependencies

4. **Install pre-commit hooks (optional but recommended):**

.. code-block:: bash

   pre-commit install

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

The development environment includes:

* **Testing**: pytest, pytest-cov, nbmake
* **Linting**: pylint, black, isort, mypy
* **Documentation**: sphinx, sphinx-rtd-theme, myst-parser
* **Profiling**: line_profiler, memory_profiler
* **Visualization**: jupyter, matplotlib, plotly

Testing Your Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Verify your setup works correctly:

.. code-block:: bash

   # Run basic tests
   python -c "import graphem as ge; print('Import successful')"
   
   # Run example scripts
   python examples/graph_generator_test.py
   python examples/real_world_datasets_test.py
   
   # Run benchmarks (takes a few minutes)
   python run_benchmarks.py

Code Style and Quality
----------------------

Coding Standards
~~~~~~~~~~~~~~~~

GraphEm follows these coding standards:

* **PEP 8** for Python style
* **Type hints** for all public functions
* **NumPy-style docstrings** for documentation
* **JAX best practices** for numerical computation
* **Consistent naming**: snake_case for functions, PascalCase for classes

Code Formatting
~~~~~~~~~~~~~~~

We use automated formatting tools:

.. code-block:: bash

   # Format code with black
   black graphem/ examples/ tests/
   
   # Sort imports with isort
   isort graphem/ examples/ tests/
   
   # Check style with pylint
   pylint graphem/

Type Checking
~~~~~~~~~~~~~

Use mypy for static type checking:

.. code-block:: bash

   mypy graphem/

Example of well-formatted code:

.. code-block:: python

   def erdos_renyi_graph(n: int, p: float, seed: int = 0) -> np.ndarray:
       """
       Generate a random undirected graph using the Erdős–Rényi G(n, p) model.

       Parameters
       ----------
       n : int
           Number of vertices.
       p : float
           Probability that an edge exists between any pair of vertices.
       seed : int, optional
           Random seed for reproducibility, by default 0.

       Returns
       -------
       np.ndarray
           Array of edge pairs (i, j) with i < j, shape (num_edges, 2).

       Examples
       --------
       >>> edges = erdos_renyi_graph(n=100, p=0.05, seed=42)
       >>> print(f"Generated {len(edges)} edges")
       """
       # Implementation here...

Testing Framework
-----------------

Test Structure
~~~~~~~~~~~~~~

GraphEm uses pytest for testing. Tests are organized as:

.. code-block::

   tests/
   ├── test_embedder.py          # Core embedding tests
   ├── test_generators.py        # Graph generator tests
   ├── test_influence.py         # Influence maximization tests
   ├── test_datasets.py          # Dataset loading tests
   ├── test_visualization.py     # Plotting and analysis tests
   └── test_benchmarks.py        # Benchmark functionality tests

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_embedder.py
   
   # Run with coverage
   pytest --cov=graphem --cov-report=html
   
   # Run performance tests (slower)
   pytest -m "not slow"

Writing Tests
~~~~~~~~~~~~~

Example test structure:

.. code-block:: python

   import pytest
   import numpy as np
   import graphem as ge


   class TestGraphEmbedder:
       """Test cases for GraphEmbedder class."""

       def test_basic_embedding(self):
           """Test basic embedding functionality."""
           edges = ge.erdos_renyi_graph(n=50, p=0.1, seed=42)
           embedder = ge.GraphEmbedder(edges=edges, n_vertices=50, dimension=2)
           
           # Test initialization
           assert embedder.n == 50
           assert embedder.dimension == 2
           
           # Test embedding computation
           embedder.run_layout(num_iterations=10)
           positions = np.array(embedder.positions)
           
           assert positions.shape == (50, 2)
           assert not np.any(np.isnan(positions))

       @pytest.mark.parametrize("dimension", [2, 3])
       def test_dimensions(self, dimension):
           """Test embedding in different dimensions."""
           edges = ge.erdos_renyi_graph(n=30, p=0.15)
           embedder = ge.GraphEmbedder(edges=edges, n_vertices=30, dimension=dimension)
           embedder.run_layout(num_iterations=5)
           
           positions = np.array(embedder.positions)
           assert positions.shape == (30, dimension)

       def test_edge_cases(self):
           """Test edge cases and error handling."""
           # Empty graph
           empty_edges = np.array([]).reshape(0, 2)
           embedder = ge.GraphEmbedder(edges=empty_edges, n_vertices=10)
           embedder.run_layout(num_iterations=1)
           
           # Single node
           single_edges = np.array([]).reshape(0, 2)
           embedder = ge.GraphEmbedder(edges=single_edges, n_vertices=1)
           embedder.run_layout(num_iterations=1)

Performance Testing
~~~~~~~~~~~~~~~~~~~

For performance-critical code, include benchmarking:

.. code-block:: python

   import time
   import pytest


   @pytest.mark.slow
   def test_large_graph_performance():
       """Test performance on large graphs."""
       edges = ge.generate_ba(n=5000, m=5)
       
       start_time = time.time()
       embedder = ge.GraphEmbedder(edges=edges, n_vertices=5000)
       embedder.run_layout(num_iterations=20)
       end_time = time.time()
       
       # Should complete within reasonable time
       assert end_time - start_time < 60  # 1 minute

Adding New Features
-------------------

Graph Generators
~~~~~~~~~~~~~~~~

When adding a new graph generator:

1. **Add to** ``graphem/generators.py``
2. **Follow the signature pattern:**

.. code-block:: python

   def generate_my_graph(n: int, param1: float, param2: int = 10, seed: int = 0) -> np.ndarray:
       """
       Generate a custom graph type.
       
       Parameters
       ----------
       n : int
           Number of vertices.
       param1 : float
           Custom parameter description.
       param2 : int, optional
           Another parameter, by default 10.
       seed : int, optional
           Random seed, by default 0.
           
       Returns
       -------
       np.ndarray
           Edge array with shape (num_edges, 2), with i < j.
       """
       np.random.seed(seed)
       
       # Generate edges ensuring i < j
       edges = []
       # ... implementation ...
       
       return np.array(edges)

3. **Update** ``graphem/__init__.py`` to export the function
4. **Add comprehensive tests**
5. **Add example to documentation**

Dataset Loaders
~~~~~~~~~~~~~~~~

For new dataset loaders:

1. **Add to** ``graphem/datasets.py``
2. **Follow the pattern:**

.. code-block:: python

   def load_my_dataset(force_download: bool = False) -> np.ndarray:
       """
       Load custom dataset.
       
       Parameters
       ----------
       force_download : bool, optional
           Force re-download even if cached, by default False.
           
       Returns
       -------
       np.ndarray
           Edge array with shape (num_edges, 2).
       """
       data_dir = get_data_directory()
       dataset_path = data_dir / "my_dataset"
       
       if not dataset_path.exists() or force_download:
           download_my_dataset(dataset_path)
       
       return load_edge_list(dataset_path / "edges.txt")

Embedding Algorithms
~~~~~~~~~~~~~~~~~~~~

For new embedding or layout algorithms:

1. **Create method in GraphEmbedder or new class**
2. **Use JAX for computations:**

.. code-block:: python

   @jit
   def my_force_function(positions: jnp.ndarray, edges: jnp.ndarray) -> jnp.ndarray:
       """JAX-compiled force computation."""
       # Use jax.numpy for all operations
       forces = jnp.zeros_like(positions)
       # ... implementation ...
       return forces

3. **Include parameters in class initialization**
4. **Add benchmarking comparison**

Documentation Guidelines
------------------------

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

All public functions must have comprehensive docstrings:

.. code-block:: python

   def complex_function(param1: int, param2: str, optional_param: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
       """
       One-line summary of the function.
       
       Longer description explaining what the function does, when to use it,
       and any important details about the implementation or algorithm.
       
       Parameters
       ----------
       param1 : int
           Description of the first parameter.
       param2 : str
           Description of the second parameter.
       optional_param : float, optional
           Description of optional parameter, by default 1.0.
           
       Returns
       -------
       Tuple[np.ndarray, Dict[str, Any]]
           Description of return values:
           - First element: array description
           - Second element: dictionary description
           
       Raises
       ------
       ValueError
           When parameter validation fails.
       RuntimeError
           When computation fails.
           
       Examples
       --------
       >>> result_array, result_dict = complex_function(10, "test")
       >>> print(result_array.shape)
       (10, 2)
       
       Notes
       -----
       Any additional notes about the algorithm, complexity, or usage.
       
       References
       ----------
       .. [1] Author, "Paper Title", Journal, Year.
       """

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   make clean  # Clean previous builds
   make html   # Build HTML documentation
   
   # View locally
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Adding Examples
~~~~~~~~~~~~~~~

When adding examples:

1. **Create complete, runnable scripts** in ``examples/``
2. **Include in documentation** with explanation
3. **Test that examples work** in CI

Pull Request Process
--------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. **Sync with upstream:**

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main

2. **Run the full test suite:**

.. code-block:: bash

   pytest
   python run_benchmarks.py
   
3. **Check code quality:**

.. code-block:: bash

   black --check graphem/
   isort --check graphem/
   pylint graphem/
   mypy graphem/

4. **Update documentation** if needed

PR Guidelines
~~~~~~~~~~~~~

* **Clear title** describing the change
* **Detailed description** explaining:
  - What the change does
  - Why it's needed
  - How it works
  - Any breaking changes
* **Link issues** if applicable
* **Include tests** for new functionality
* **Update documentation** for user-facing changes

Example PR template:

.. code-block:: markdown

   ## Summary
   Brief description of what this PR does.

   ## Changes
   - Add new graph generator for hierarchical networks
   - Update documentation with examples
   - Add comprehensive tests

   ## Testing
   - [ ] All existing tests pass
   - [ ] New tests added and passing
   - [ ] Benchmarks run successfully
   - [ ] Documentation builds without errors

   ## Breaking Changes
   None / List any breaking changes

Benchmarking Changes
~~~~~~~~~~~~~~~~~~~~

For performance-related changes, include benchmarks:

.. code-block:: bash

   # Run before changes
   python run_benchmarks.py > benchmark_before.txt
   
   # Make changes
   # ...
   
   # Run after changes  
   python run_benchmarks.py > benchmark_after.txt
   
   # Compare results
   diff benchmark_before.txt benchmark_after.txt

Release Process
---------------

GraphEm follows semantic versioning (MAJOR.MINOR.PATCH):

* **MAJOR**: Breaking API changes
* **MINOR**: New features, backwards compatible
* **PATCH**: Bug fixes, backwards compatible

For maintainers, the release process involves:

1. Update version in ``setup.py`` and ``__init__.py``
2. Update ``CHANGELOG.md``
3. Create release tag
4. Build and upload to PyPI
5. Update documentation

Getting Help
------------

If you need help contributing:

1. **Check existing issues** on GitHub
2. **Ask questions** in issue discussions
3. **Join our community** (links in README)
4. **Read the code** - GraphEm is designed to be readable

**Communication Guidelines:**

* Be respectful and constructive
* Provide context and examples
* Search before asking duplicate questions
* Help others when you can

Thank you for contributing to GraphEm! 🚀