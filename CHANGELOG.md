# Changelog

All notable changes to GraphEm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Critical bug fix in influence maximization** (`graphem/influence.py`): Fixed `ndlib_estimated_influence()` function to correctly initialize seed nodes using NDlib's proper API. Previously, the function was using an incorrect configuration method that resulted in seeds not being properly set, leading to inaccurate influence estimations. The fix ensures:
  - Seeds are now correctly initialized using `config.add_model_initial_configuration("Infected", seeds)` instead of the incorrect `config.add_node_configuration("status", seed, 1)`
  - Influenced node counts are now correctly retrieved from `iterations[-1]['node_count'].get(2, 0)` instead of manually iterating through status values
  - All influence maximization benchmarks and comparisons now produce accurate results
  - High-degree seeds now correctly show higher influence than low-degree seeds

This bug affected all influence maximization functionality including `graphem_seed_selection()`, `greedy_seed_selection()`, and benchmark comparisons. Users should re-run any influence maximization experiments performed with previous versions.

## [Previous Releases]

For release history before this changelog was established, see the [GitHub Releases](https://github.com/igorrivin/graphem/releases) page.
