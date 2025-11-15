"""Unit tests for graph generators."""

import pytest
import numpy as np
import networkx as nx
import scipy.sparse as sp
from graphem.generators import (
    generate_er,
    generate_random_regular,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_relaxed_caveman,
    generate_ws,
    generate_ba,
    generate_sbm,
    generate_bipartite_graph,
    generate_complete_bipartite_graph,
    generate_delaunay_triangulation,
    compute_vertex_degrees
)


class TestGenerators:
    """Test graph generators."""

    def test_erdos_renyi_graph(self):
        """Test Erdős-Rényi graph generator."""
        n, p = 50, 0.1
        adj = generate_er(n=n, p=p, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)
        assert adj.dtype == np.int32 or adj.dtype == np.int64

        # Check symmetric
        assert np.allclose((adj - adj.T).data, 0)

        # Check no self-loops
        assert adj.diagonal().sum() == 0

        # Check edges within range
        assert 0 <= adj.nnz // 2 <= n * (n - 1) // 2

    def test_random_regular_graph(self):
        """Test random regular graph generator."""
        n, d = 20, 3
        adj = generate_random_regular(n=n, d=d, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)

        # Check all vertices have degree d
        degrees = compute_vertex_degrees(adj)
        assert np.all(degrees == d)

    def test_scale_free_graph(self):
        """Test scale-free graph generator."""
        n = 50
        adj = generate_scale_free(n=n, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)
        assert adj.nnz > 0

    def test_geometric_graph(self):
        """Test random geometric graph generator."""
        n, radius = 30, 0.3
        adj = generate_geometric(n=n, radius=radius, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)

    def test_caveman_graph(self):
        """Test caveman graph generator."""
        l, k = 3, 5
        adj = generate_caveman(l=l, k=k)

        assert sp.issparse(adj)
        total_vertices = l * k
        assert adj.shape == (total_vertices, total_vertices)

    def test_relaxed_caveman_graph(self):
        """Test relaxed caveman graph generator."""
        l, k, p = 3, 5, 0.1
        adj = generate_relaxed_caveman(l=l, k=k, p=p, seed=42)

        assert sp.issparse(adj)
        total_vertices = l * k
        assert adj.shape == (total_vertices, total_vertices)

    def test_watts_strogatz_graph(self):
        """Test Watts-Strogatz small-world graph generator."""
        n, k, p = 20, 4, 0.3
        adj = generate_ws(n=n, k=k, p=p, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)

    def test_barabasi_albert_graph(self):
        """Test Barabási-Albert graph generator."""
        n, m = 50, 2
        adj = generate_ba(n=n, m=m, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)

    def test_stochastic_block_model(self):
        """Test Stochastic Block Model generator."""
        n_per_block, num_blocks = 10, 3
        p_in, p_out = 0.8, 0.1
        adj = generate_sbm(
            n_per_block=n_per_block,
            num_blocks=num_blocks,
            p_in=p_in,
            p_out=p_out,
            seed=42
        )

        assert sp.issparse(adj)
        total_vertices = n_per_block * num_blocks
        assert adj.shape == (total_vertices, total_vertices)

    def test_bipartite_graph(self):
        """Test random bipartite graph generator."""
        n_top, n_bottom = 20, 30
        p = 0.2
        adj = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=p, seed=42)

        assert sp.issparse(adj)
        total_vertices = n_top + n_bottom
        assert adj.shape == (total_vertices, total_vertices)

        # Test reproducibility with same seed
        adj2 = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=p, seed=42)
        assert np.allclose((adj - adj2).data, 0)

        # Test different seeds give different results
        adj3 = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=p, seed=123)
        if adj.nnz > 0 and adj3.nnz > 0:
            assert not np.allclose((adj - adj3).data, 0)

    def test_complete_bipartite_graph(self):
        """Test complete bipartite graph generator."""
        n_top, n_bottom = 10, 15
        adj = generate_complete_bipartite_graph(n_top=n_top, n_bottom=n_bottom)

        assert sp.issparse(adj)
        total_vertices = n_top + n_bottom
        assert adj.shape == (total_vertices, total_vertices)

        # Complete bipartite should have exactly n_top * n_bottom edges
        assert adj.nnz == 2 * n_top * n_bottom  # Each edge counted twice

        # Convert to NetworkX and verify bipartite structure
        G = nx.from_scipy_sparse_array(adj)
        assert nx.is_bipartite(G)

    def test_delaunay_triangulation(self):
        """Test Delaunay triangulation graph generator."""
        n = 50
        adj = generate_delaunay_triangulation(n=n, seed=42)

        assert sp.issparse(adj)
        assert adj.shape == (n, n)
        assert adj.nnz > 0

        # Convert to NetworkX
        G = nx.from_scipy_sparse_array(adj)

        # Delaunay triangulation should be connected for random points
        assert nx.is_connected(G)

        # Test reproducibility with same seed
        adj2 = generate_delaunay_triangulation(n=n, seed=42)
        assert np.allclose((adj - adj2).data, 0)

        # Test different seeds give different results
        adj3 = generate_delaunay_triangulation(n=n, seed=123)
        assert not np.allclose((adj - adj3).data, 0)

    def test_reproducible_results(self):
        """Test that generators produce reproducible results with same seed."""
        n, p = 30, 0.2

        adj1 = generate_er(n=n, p=p, seed=123)
        adj2 = generate_er(n=n, p=p, seed=123)

        assert np.allclose((adj1 - adj2).data, 0)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        n, p = 30, 0.3

        adj1 = generate_er(n=n, p=p, seed=123)
        adj2 = generate_er(n=n, p=p, seed=456)

        if adj1.nnz > 0 and adj2.nnz > 0:
            assert not np.allclose((adj1 - adj2).data, 0)

    def test_adjacency_format(self):
        """Test that all generators return sparse matrices in consistent format."""
        generators_params = [
            (generate_er, {"n": 20, "p": 0.1, "seed": 42}),
            (generate_random_regular, {"n": 20, "d": 3, "seed": 42}),
            (generate_ws, {"n": 20, "k": 4, "p": 0.3, "seed": 42}),
            (generate_ba, {"n": 20, "m": 2, "seed": 42}),
        ]

        for generator, params in generators_params:
            adj = generator(**params)

            assert sp.issparse(adj)
            assert adj.shape[0] == adj.shape[1]  # Square
            assert adj.shape[0] > 0  # Non-empty
            # Check symmetric (undirected graph)
            assert np.allclose((adj - adj.T).data, 0)

    def test_compute_vertex_degrees(self):
        """Test vertex degree computation."""
        # Create a simple graph
        n = 5
        adj = generate_er(n=n, p=0.5, seed=42)

        degrees = compute_vertex_degrees(adj)

        assert isinstance(degrees, np.ndarray)
        assert degrees.shape == (n,)
        assert np.all(degrees >= 0)

        # Verify degree calculation
        adj_csr = sp.csr_matrix(adj)  # Convert to csr_matrix for row slicing
        for i in range(n):
            expected_degree = adj_csr[i].nnz
            assert degrees[i] == expected_degree
