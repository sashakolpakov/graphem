"""
A JAX-based implementation of graph embedding.
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import plotly.graph_objects as go
from loguru import logger
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

from graphem.index import HPIndex


class GraphEmbedder:
    """
    A class for embedding graphs using the Laplacian embedding.

    Attributes:
            adjacency: scipy.sparse matrix
                Sparse adjacency matrix of the graph.
            n: int
                Number of vertices in the graph.
            n_components: int
                Dimension of the embedding.
            L_min: float
                Minimum length of the spring.
            k_attr: float
                Attraction force constant.
            k_inter: float
                Repulsion force constant for intersections.
            n_neighbors: int
                Number of nearest neighbors to consider.
            sample_size: int
                Number of samples for kNN search (automatically limited to number of edges).
            batch_size: int
                Batch size for kNN search (automatically limited to number of vertices).
            logger_instance: loguru.logger
                Logger object to use for logging.
    """
    def __init__(self, adjacency, n_components=2, L_min=1.0, k_attr=0.2, k_inter=0.5, n_neighbors=10, sample_size=256, batch_size=1024, logger_instance=None, verbose=True, seed=None):
        """
        Initialize the GraphEmbedder.

        Parameters:
            adjacency: scipy.sparse matrix
                Sparse adjacency matrix of the graph.
            n_components: int, default=2
                Dimension of the embedding.
            L_min: float, default=1.0
                Minimum length of the spring.
            k_attr: float, default=0.2
                Attraction force constant.
            k_inter: float, default=0.5
                Repulsion force constant for intersections.
            n_neighbors: int, default=10
                Number of nearest neighbors to consider.
            sample_size: int, default=256
                Number of samples for kNN search.
            batch_size: int, default=1024
                Batch size for kNN search.
            logger_instance: loguru.logger, optional
                Logger object to use for logging.
            verbose: bool, default=True
                Whether to display progress information.
            seed: int, optional
                Random seed for reproducibility.
        """
        # Validate and convert adjacency matrix to csr_matrix
        if sp.issparse(adjacency):
            # Convert to csr_matrix (not csr_array) for compatibility
            adjacency = sp.csr_matrix(adjacency)
        elif isinstance(adjacency, np.ndarray):
            adjacency = sp.csr_matrix(adjacency)
        else:
            adjacency = sp.csr_matrix(np.asarray(adjacency))

        # Check if square
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}")

        # Check for empty graph
        if adjacency.shape[0] == 0:
            raise ValueError("Adjacency matrix cannot be empty")

        self.adjacency = adjacency
        self.n = adjacency.shape[0]

        # Extract edges from adjacency matrix for JAX operations
        edges_coo = sp.triu(self.adjacency, k=1).tocoo()
        edges = np.column_stack([edges_coo.row, edges_coo.col])
        self.edges = jnp.array(edges)

        self.n_components = n_components
        self.L_min = L_min
        self.k_attr = k_attr
        self.k_inter = k_inter
        self.n_neighbors = n_neighbors
        self.sample_size = min(sample_size, len(edges))
        self.batch_size = min(batch_size, self.n)
        self.seed = seed

        if logger_instance is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w', encoding='utf-8')
            logger.add(sink, level="INFO")
            self.logger = logger
        else:
            self.logger = logger_instance
            self.logger.info("Logger initialized")

        self._positions = self._laplacian_embedding()

    @property
    def positions(self):
        """Get the current positions of vertices."""
        return self._positions

    def get_positions(self):
        """
        Get the current positions of vertices.

        Returns:
            np.ndarray: Array of shape (n, n_components) with vertex positions.
        """
        return np.array(self._positions)

    def _laplacian_embedding(self):
        """
        Compute the Laplacian embedding of the graph.
        """
        self.logger.info("Computing Laplacian embedding")
        L = laplacian(self.adjacency, normed=True)
        k = self.n_components + 1
        _, eigenvectors = spla.eigsh(L, k, which='SM')
        lap_embedding = eigenvectors[:, 1:k]
        self.logger.info("Laplacian embedding done")
        return jnp.array(lap_embedding)

    def locate_knn_midpoints(self, midpoints, k):
        """
        Locate k nearest neighbors for each midpoint.
        """
        self.logger.info("Locating kNN midpoints")
        E = midpoints.shape[0]

        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, E, shape=(self.sample_size,), replace=False)

        # Move to numpy for slicing to avoid dynamic tracing overhead
        idx_np = np.array(idx)
        sampled_midpoints = midpoints[idx_np]

        jax_indices, jax_distances = HPIndex.knn_tiled(
            sampled_midpoints,  # batch
            midpoints,  # full data
            k + 1,
            self.sample_size,
            self.batch_size
        )
        jax_indices.block_until_ready()
        jax_distances.block_until_ready()
        self.logger.info("kNN midpoints done")
        return jax_indices[:, 1:], idx

    @staticmethod
    @jit
    def compute_spring_forces(positions, edges, L_min, k_attr):
        """
        Compute the spring forces between vertices.
        """
        p1 = positions[edges[:, 0]]
        p2 = positions[edges[:, 1]]
        diff = p2 - p1
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        force_magnitude = -k_attr * (dist - L_min)
        edge_force = force_magnitude * (diff / dist)
        forces = jnp.zeros_like(positions)
        forces = forces.at[edges[:, 0]].add(edge_force)
        forces = forces.at[edges[:, 1]].add(-edge_force)
        return forces

    @staticmethod
    @jit
    def compute_intersection_forces_with_knn_index(positions, edges, knn_idx, sampled_indices, k_inter):
        """
        Compute the intersection forces between vertices and their nearest neighbors.
        """
        row_idx = jnp.arange(knn_idx.shape[0])
        candidate_i = sampled_indices[row_idx.repeat(knn_idx.shape[1])]
        candidate_j = knn_idx.flatten()

        valid_mask = candidate_i < candidate_j

        edges_i = edges[candidate_i]
        edges_j = edges[candidate_j]

        share_mask = ((edges_i[:, 0] == edges_j[:, 0]) |
                      (edges_i[:, 0] == edges_j[:, 1]) |
                      (edges_i[:, 1] == edges_j[:, 0]) |
                      (edges_i[:, 1] == edges_j[:, 1]))
        overall_valid = valid_mask & (~share_mask)

        p1 = positions[edges_i[:, 0]]
        p2 = positions[edges_i[:, 1]]
        q1 = positions[edges_j[:, 0]]
        q2 = positions[edges_j[:, 1]]

        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
                   (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)
        intersect_mask = (o1 * o2 < 0) & (o3 * o4 < 0)
        final_mask = overall_valid & intersect_mask
        final_mask_float = final_mask.astype(jnp.float32)[:, None]

        inter_midpoints = (p1 + p2 + q1 + q2) / 4.0

        def repulse(vertex, mid):
            d = jnp.linalg.norm(vertex - mid) + 1e-6
            return k_inter * (vertex - mid) / (d ** 2)

        repulse_v = vmap(repulse, in_axes=(0, 0))
        force_i0 = repulse_v(positions[edges_i[:, 0]], inter_midpoints)
        force_i1 = repulse_v(positions[edges_i[:, 1]], inter_midpoints)
        force_j0 = repulse_v(positions[edges_j[:, 0]], inter_midpoints)
        force_j1 = repulse_v(positions[edges_j[:, 1]], inter_midpoints)

        force_i0 *= final_mask_float
        force_i1 *= final_mask_float
        force_j0 *= final_mask_float
        force_j1 *= final_mask_float

        forces = jnp.zeros_like(positions)
        forces = forces.at[edges_i[:, 0]].add(force_i0)
        forces = forces.at[edges_i[:, 1]].add(force_i1)
        forces = forces.at[edges_j[:, 0]].add(force_j0)
        forces = forces.at[edges_j[:, 1]].add(force_j1)

        return forces

    def update_positions(self):
        """
        Update the positions of the vertices based on the spring forces and intersection forces.
        """
        self.logger.info("Updating positions")
        spring_forces = self.compute_spring_forces(self._positions, self.edges, self.L_min, self.k_attr)
        midpoints = (self._positions[self.edges[:, 0]] + self._positions[self.edges[:, 1]]) / 2.0
        knn_idx, sampled_indices = self.locate_knn_midpoints(midpoints, self.n_neighbors)
        inter_forces = self.compute_intersection_forces_with_knn_index(self._positions, self.edges, knn_idx, sampled_indices, self.k_inter)
        forces = spring_forces + inter_forces
        new_positions = self._positions + forces
        self._positions = (new_positions - jnp.mean(new_positions, axis=0)) / (jnp.std(new_positions, axis=0) + 1e-6)
        self.logger.info("Positions updated")

    def run_layout(self, num_iterations=100):
        """
        Run the layout for a given number of iterations.
        """
        self.logger.info("Running layout")
        for _ in tqdm(range(num_iterations)):
            self.update_positions()
        return self.get_positions()

    def display_layout(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display the graph embedding using Plotly.


        Parameters
        ----------
        edge_width : float
            The width of the edges in the graph embedding.
        node_size : float
            The size of the nodes in the graph embedding.
        node_colors : array-like of shape (num_vertices,)
            An array of colors for each vertex.

        Returns
        -------
        None
            Displays the graph embedding using Plotly in the appropriate dimension.
        """
        self.logger.info("Displaying layout")
        if self.n_components == 2:
            self._display_layout_2d(edge_width=edge_width, node_size=node_size, node_colors=node_colors)
        elif self.n_components == 3:
            self._display_layout_3d(edge_width=edge_width, node_size=node_size, node_colors=node_colors)
        else:
            raise ValueError("Dimension must be 2 or 3 to display layout")

    def _display_layout_2d(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display a 2D graph embedding using Plotly.


        Parameters
        ----------
        edge_width : float
            The width of the edges in the graph embedding.
        node_size : float
            The size of the nodes in the graph embedding.
        node_colors : array-like of shape (num_vertices,)
            An array of colors for each vertex.

        Returns
        -------
        None
            Displays a Plotly 2D figure with vertices plotted as red markers and edges as gray lines.
        """
        pos = np.array(self._positions)
        edges = np.array(self.edges)

        x_edges = []
        y_edges = []
        for i, j in edges:
            x_edges += [pos[i, 0], pos[j, 0], None]
            y_edges += [pos[i, 1], pos[j, 1], None]

        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line={"color": 'gray', "width": edge_width},
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode='markers',
            marker={
                "color": node_colors if node_colors is not None else 'red',
                "colorscale": 'Bluered',
                "size": node_size,
                "colorbar": {"title": 'Node Label'},
                "showscale": node_colors is not None
            },
            hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="2D Graph Embedding",
            xaxis={"title": 'X', "showgrid": False, "zeroline": False},
            yaxis={"title": 'Y', "showgrid": False, "zeroline": False},
            showlegend=False,
            width=800,
            height=800
        )
        fig.show()

    def _display_layout_3d(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display a 3D graph embedding using Plotly.


        Parameters
        ----------
        edge_width : float
            The width of the edges in the graph embedding.
        node_size : float
            The size of the nodes in the graph embedding.
        node_colors : array-like of shape (num_vertices,)
            An array of colors for each vertex.

        Returns
        -------
        None
            Displays a Plotly 3D figure with vertices plotted as red markers and edges as gray lines.
        """
        pos = np.array(self._positions)
        edges = np.array(self.edges)

        x_edges, y_edges, z_edges = [], [], []
        for i, j in edges:
            x_edges += [pos[i, 0], pos[j, 0], None]
            y_edges += [pos[i, 1], pos[j, 1], None]
            z_edges += [pos[i, 2], pos[j, 2], None]

        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line={"color": 'gray', "width": edge_width},
            hoverinfo='none'
        )

        node_trace = go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker={
                "color": node_colors if node_colors is not None else 'red',
                "colorscale": 'Bluered',
                "size": node_size,
                "colorbar": {"title": 'Node Label'},
                "showscale": node_colors is not None
            },
            hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Graph Embedding",
            scene={"xaxis": {"title": 'X'}, "yaxis": {"title": 'Y'}, "zaxis": {"title": 'Z'}},
            showlegend=False,
            width=800,
            height=800
        )
        fig.show()
