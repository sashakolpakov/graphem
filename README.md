<p align="center">
  <img src="docs/logo.png" alt="GraphEm logo" height="240">
</p>

<h1 align="center">GraphEm: geometric graph embedding and radial node ranking</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT license"></a>
  <a href="https://pypi.org/project/graphem-jax/"><img src="https://img.shields.io/pypi/v/graphem-jax.svg" alt="PyPI version"></a>
  <a href="https://github.com/sashakolpakov/graphem/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/tests.yml?branch=main&label=tests&logo=github" alt="Tests"></a>
  <a href="https://github.com/sashakolpakov/graphem/actions/workflows/deploy_docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/graphem/deploy_docs.yml?branch=main&label=docs&logo=github" alt="Documentation"></a>
  <a href="https://doi.org/10.21105/joss.08855"><img src="https://joss.theoj.org/papers/10.21105/joss.08855/status.svg" alt="JOSS paper"></a>
</p>

GraphEm constructs a low-dimensional graph layout and uses each vertex's radial
distance as a ranking score. This repository contains the portable JAX reference
package, published as `graphem-jax`. The production CUDA implementation and its
large-graph benchmark harness live in
[`graphem-rapids`](https://github.com/sashakolpakov/graphem-rapids).

The radial score is an empirical proxy, not an exact centrality or influence
oracle. Evaluate it against task-appropriate baselines on the graph family that
matters to you.

## Features

- normalized-Laplacian initialization followed by force-directed refinement;
- sparse-adjacency input and graph-generator helpers;
- radial node ranking and centrality-correlation reports;
- optional Independent-Cascade evaluation through NDlib;
- SNAP and Network Repository dataset helpers;
- Plotly visualization; and
- a separate CUDA implementation for production-scale runs.

## Installation

Install the current `0.2.x` API from its release tag:

```bash
python -m pip install "graphem-jax @ git+https://github.com/sashakolpakov/graphem.git@graphem-jax-0.2.0"
```

PyPI currently carries the legacy `0.1.0` package. These pages track the
repository's `0.2.x` adjacency-object API, so a plain `pip install graphem-jax`
does not yet match the examples below. Check `graphem.__version__` when
reproducing an older environment.

Install a development checkout:

```bash
git clone https://github.com/sashakolpakov/graphem.git
cd graphem
python -m pip install -e ".[test,docs]"
```

JAX accelerator wheels are platform-specific. Follow the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
when using this package on an accelerator. For the H100-qualified CUDA path, use
[`graphem-rapids`](https://github.com/sashakolpakov/graphem-rapids).

## Quick start

```python
import graphem as ge

adjacency = ge.generate_er(n=500, p=0.01, seed=7)
embedder = ge.GraphEmbedder(
    adjacency=adjacency,
    n_components=3,
    seed=7,
    verbose=False,
)
positions = embedder.run_layout(num_iterations=50)
scores = (positions**2).sum(axis=1) ** 0.5
top_nodes = scores.argsort()[::-1][:10]
print(top_nodes)
```

`GraphEmbedder` accepts a square dense or SciPy sparse adjacency object. Graph
generators in `graphem.generators` return SciPy CSR sparse objects.

## Influence evaluation

`graphem_seed_selection` selects the vertices with the largest radial scores.
Use `ndlib_estimated_influence` to evaluate a fixed seed set and
`greedy_seed_selection` as a small-graph baseline:

```python
import networkx as nx
import graphem as ge

adjacency = ge.generate_er(n=128, p=0.05, seed=3)
graph = nx.from_scipy_sparse_array(adjacency)
embedder = ge.GraphEmbedder(adjacency, seed=3, verbose=False)

radial_seeds = ge.graphem_seed_selection(embedder, k=10, num_iterations=20)
spread, simulated_steps = ge.ndlib_estimated_influence(
    graph,
    radial_seeds,
    p=0.1,
    iterations_count=200,
)
print(spread, simulated_steps)
```

Influence estimates are stochastic. Compare methods on shared propagation worlds
or sufficiently large independent samples; do not interpret one NDlib trajectory
as a method comparison.

## Benchmarking

```python
from graphem.benchmark import benchmark_correlations
from graphem.generators import generate_er

result = benchmark_correlations(
    generate_er,
    graph_params={"n": 200, "p": 0.05, "seed": 11},
    n_components=3,
    num_iterations=40,
)
print(result)
```

Large CUDA reproduction runs are content-addressed per cell so that a corrected
or extended cell can be independently rerun and audited without replacing
unrelated evidence.

## Examples

- `examples/graph_generator_example.py`
- `examples/real_world_datasets_example.py`
- `examples/graphem_jax_notebook.ipynb` — an output-free, CPU-safe walkthrough
  of the current adjacency-object API and full-ranking Spearman evaluation

## Development

```bash
python -m pytest tests -v
python build_docs.py
```

`build_docs.py` treats every Sphinx warning as an error. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the complete contributor workflow.

## Documentation

The deployed Sphinx documentation is at
[sashakolpakov.github.io/graphem](https://sashakolpakov.github.io/graphem/).

## Citation

The manuscript and its version history are available as
[arXiv:2506.07435](https://arxiv.org/abs/2506.07435).

```bibtex
@misc{kolpakov-rivin-2025fast,
  title        = {Fast Geometric Embedding for Node Influence Maximization},
  author       = {Kolpakov, Alexander and Rivin, Igor},
  year         = {2025},
  eprint       = {2506.07435},
  archivePrefix= {arXiv},
  primaryClass = {cs.SI},
  url          = {https://arxiv.org/abs/2506.07435}
}
```

## License

[MIT](LICENSE)
