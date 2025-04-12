#!/usr/bin/env python
# coding: utf-8

# # Installing packages

# In[1]:


get_ipython().system('pip install -U ndlib loguru kaleido')


# # Kernelized kNN search

# In[2]:


"""
A JAX-based implementation for efficient k-nearest neighbors.
"""

from functools import partial
import jax
import jax.numpy as jnp


class HPIndex:

    """
    A kernelized kNN index that uses batching / tiling to efficiently handle
    large datasets with limited memory usage.
    """

    def __init__(self):
        pass

    @staticmethod
    def knn_tiled(x, y, k=5, x_tile_size=8192, y_batch_size=1024):
        """
        Advanced implementation that tiles both database and query points.
        This wrapper handles the dynamic aspects before calling the JIT-compiled
        function.

        Args:
            x: (n, d) array of database points
            y: (m, d) array of query points
            k: number of nearest neighbors
            x_tile_size: size of database tiles
            y_batch_size: size of query batches

        Returns:
            (m, k) array of indices of nearest neighbors
        """
        n_x, _ = x.shape
        n_y, _ = y.shape

        # Ensure batch sizes aren't larger than the data dimensions
        x_tile_size = min(x_tile_size, n_x)
        y_batch_size = min(y_batch_size, n_y)

        # Calculate batching parameters
        num_y_batches = n_y // y_batch_size
        y_remainder = n_y % y_batch_size
        num_x_tiles = (n_x + x_tile_size - 1) // x_tile_size

        # Call the JIT-compiled implementation with concrete values
        return HPIndex._knn_tiled_jit(
            x, y, k, x_tile_size, y_batch_size,
            num_y_batches, y_remainder, num_x_tiles, n_x
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
    def _knn_tiled_jit(x, y, k, x_tile_size, y_batch_size,
                       num_y_batches, y_remainder, num_x_tiles, n_x):
        """
        JIT-compiled implementation of tiled KNN with concrete batch parameters.
        """
        n_y, d_y = y.shape
        _, d_x = x.shape

        # Initialize results
        all_indices = jnp.zeros((n_y, k), dtype=jnp.int32)
        all_distances = jnp.ones((n_y, k)) * jnp.finfo(jnp.float32).max

        # Define the scan function for processing y batches
        def process_y_batch(carry, y_batch_idx):
            curr_indices, curr_distances = carry

            # Get current batch of query points
            y_start = y_batch_idx * y_batch_size
            y_batch = jax.lax.dynamic_slice(y, (y_start, 0), (y_batch_size, d_y))

            # Initialize batch results
            batch_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            batch_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            # Define the scan function for processing x tiles within a y batch
            def process_x_tile(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use a fixed size for the slice and then mask invalid values
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate how many elements are actually valid
                # (This is now done without dynamic shapes)
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between y_batch and x_tile
                tile_distances = _compute_batch_distances(y_batch, x_tile)

                # Mask out invalid indices (those beyond the actual data)
                valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                # Adjust indices to account for tile offset
                # Make sure indices are within bounds
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for this y batch
            (batch_indices, batch_distances), _ = jax.lax.scan(
                process_x_tile,
                (batch_indices, batch_distances),
                jnp.arange(num_x_tiles)
            )

            # Update overall results for this batch
            curr_indices = jax.lax.dynamic_update_slice(
                curr_indices, batch_indices, (y_start, 0)
            )
            curr_distances = jax.lax.dynamic_update_slice(
                curr_distances, batch_distances, (y_start, 0)
            )

            return (curr_indices, curr_distances), None

        # Process all full y batches
        (all_indices, all_distances), _ = jax.lax.scan(
            process_y_batch,
            (all_indices, all_distances),
            jnp.arange(num_y_batches)
        )

        # Handle y remainder with similar changes if needed
        def handle_y_remainder(indices, distances):
            y_start = num_y_batches * y_batch_size

            # Get and pad remainder batch
            remainder_y = jax.lax.dynamic_slice(y, (y_start, 0), (y_remainder, d_y))
            padded_y = jnp.pad(remainder_y, ((0, y_batch_size - y_remainder), (0, 0)))

            # Initialize remainder results
            remainder_indices = jnp.zeros((y_batch_size, k), dtype=jnp.int32)
            remainder_distances = jnp.ones((y_batch_size, k)) * jnp.finfo(jnp.float32).max

            # Process x tiles for the remainder batch (with same fix as above)
            def process_x_tile_remainder(carry, x_tile_idx):
                batch_idx, batch_dist = carry

                # Get current tile of database points - use fixed size slices
                x_start = x_tile_idx * x_tile_size

                # Use fixed size for the slice
                x_tile = jax.lax.dynamic_slice(
                    x, (x_start, 0), (x_tile_size, d_x)
                )

                # Calculate actual valid size
                x_tile_actual_size = jnp.minimum(x_tile_size, n_x - x_start)

                # Compute distances between padded_y and x_tile
                tile_distances = _compute_batch_distances(padded_y, x_tile)

                # Mask out invalid indices (both for y padding and x overflow)
                x_valid_mask = jnp.arange(x_tile_size) < x_tile_actual_size
                tile_distances = jnp.where(
                    x_valid_mask[jnp.newaxis, :],
                    tile_distances,
                    jnp.ones_like(tile_distances) * jnp.finfo(jnp.float32).max
                )

                # Adjust indices to account for tile offset
                tile_indices = jnp.minimum(
                    jnp.arange(x_tile_size) + x_start,
                    n_x - 1  # Ensure indices don't go beyond n_x
                )
                tile_indices = jnp.broadcast_to(tile_indices, tile_distances.shape)

                # Merge current tile results with previous results
                combined_distances = jnp.concatenate([batch_dist, tile_distances], axis=1)
                combined_indices = jnp.concatenate([batch_idx, tile_indices], axis=1)

                # Sort and get top k
                top_k_idx = jnp.argsort(combined_distances)[:, :k]

                # Gather top k distances and indices
                new_batch_dist = jnp.take_along_axis(combined_distances, top_k_idx, axis=1)
                new_batch_idx = jnp.take_along_axis(combined_indices, top_k_idx, axis=1)

                return (new_batch_idx, new_batch_dist), None

            # Process all x tiles for the remainder batch
            (remainder_indices, remainder_distances), _ = jax.lax.scan(
                process_x_tile_remainder,
                (remainder_indices, remainder_distances),
                jnp.arange(num_x_tiles)
            )

            # Extract valid remainder results and update
            valid_remainder_indices = remainder_indices[:y_remainder]

            indices = jax.lax.dynamic_update_slice(
                indices, valid_remainder_indices, (y_start, 0)
            )

            return indices, distances

        # Conditionally handle remainder to avoid issues with remainder=0
        all_indices, all_distances = jax.lax.cond(
            y_remainder > 0,
            lambda args: handle_y_remainder(*args),
            lambda args: args,
            (all_indices, all_distances)
        )

        return all_indices, all_distances


# Globally define the _compute_batch_distances function for reuse
@jax.jit
def _compute_batch_distances(y_batch, x):
    """
    Compute the squared distances between a batch of query points and all
    database points.

    Args:
        y_batch: (batch_size, d) array of query points
        x: (n, d) array of database points

    Returns:
        (batch_size, n) array of squared distances
    """
    # Compute squared norms
    x_norm = jnp.sum(x**2, axis=1)
    y_norm = jnp.sum(y_batch**2, axis=1)

    # Compute xy term
    xy = jnp.dot(y_batch, x.T)

    # Complete squared distance: ||y||² + ||x||² - 2*<y,x>
    dists2 = y_norm[:, jnp.newaxis] + x_norm[jnp.newaxis, :] - 2 * xy
    dists2 = jnp.clip(dists2, 0, jnp.finfo(jnp.float32).max)

    return dists2


# # Graph Embedding

# In[3]:


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

class GraphEmbedder:
    """
    A class for embedding graphs using the Laplacian embedding.

    Attributes:
            edges: np.ndarray of shape (num_edges, 2)
                Array of edge pairs (i, j) with i < j.
            n_vertices: int
                Number of vertices in the graph.
            dimension: int
                Dimension of the embedding.
            L_min: float
                Minimum length of the spring.
            k_attr: float
                Attraction force constant.
            k_inter: float
                Repulsion force constant for intersections.
            knn_k: int
                Number of nearest neighbors to consider.
            sample_size: int
                Number of samples for kNN search.
            batch_size: int
                Batch size for kNN search.
            my_logger: loguru.logger
                Logger object to use for logging.
    """
    def __init__(self, edges, n_vertices, dimension=2, L_min=1.0, k_attr=0.2, k_inter=0.5, knn_k=10, sample_size=256, batch_size=1024, my_logger=None, verbose=True):
        """
        Initialize the GraphEmbedder.
        """
        self.edges = jnp.array(edges)
        self.n = n_vertices
        self.dimension = dimension
        self.L_min = L_min
        self.k_attr = k_attr
        self.k_inter = k_inter
        self.knn_k = knn_k
        self.sample_size = sample_size
        self.batch_size = batch_size
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w', encoding='utf-8')
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger
            self.logger.info("Logger initialized")
        self.positions = self._laplacian_embedding()

    def _laplacian_embedding(self):
        """
        Compute the Laplacian embedding of the graph.
        """
        self.logger.info("Computing Laplacian embedding")
        edges_np = np.array(self.edges)
        row = edges_np[:, 0]
        col = edges_np[:, 1]
        data = np.ones(len(edges_np))
        A = sp.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        L = laplacian(A + A.transpose(), normed=True)
        k = self.dimension + 1
        _, eigenvectors = spla.eigsh(L, k, which='SM')
        lap_embedding = eigenvectors[:, 1:k]
        self.logger.info("Laplacian embedding done")
        return jnp.array(lap_embedding)


    def locate_knn_midpoints(self, midpoints, k):
        """
        Locate k nearest neighbors for each midpoint.
        """
        self.logger.info("Locating knn midpoints")
        E = midpoints.shape[0]
        idx = jax.random.choice(jax.random.PRNGKey(0), E, shape=(self.sample_size,), replace=False)
        jax_indices, jax_distances = HPIndex.knn_tiled(midpoints[idx], midpoints, k+1, self.sample_size, self.batch_size)
        jax_indices.block_until_ready()
        jax_distances.block_until_ready()
        self.logger.info("Knn midpoints done")
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
        spring_forces = self.compute_spring_forces(self.positions, self.edges, self.L_min, self.k_attr)
        midpoints = (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]) / 2.0
        knn_idx, sampled_indices = self.locate_knn_midpoints(midpoints, self.knn_k)
        inter_forces = self.compute_intersection_forces_with_knn_index(self.positions, self.edges, knn_idx, sampled_indices, self.k_inter)
        forces = spring_forces + inter_forces
        new_positions = self.positions + forces
        self.positions = (new_positions - jnp.mean(new_positions, axis=0)) / (jnp.std(new_positions, axis=0) + 1e-6)
        self.logger.info("Positions updated")

    def run_layout(self, num_iterations=100):
        """
        Run the layout for a given number of iterations.
        """
        self.logger.info("Running layout")
        for _ in tqdm(range(num_iterations)):
            self.update_positions()
        return self.positions

    def display_layout(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display the graph embedding using Plotly.
        """
        self.logger.info("Displaying layout")
        if self.dimension == 2:
            self._display_layout_2d(edge_width=edge_width, node_size=node_size, node_colors=node_colors)
        elif self.dimension == 3:
            self._display_layout_3d(edge_width=edge_width, node_size=node_size, node_colors=node_colors)
        else:
            raise ValueError("Dimension must be 2 or 3 to display layout")

    def _display_layout_2d(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display a 2D graph embedding using Plotly.

        Parameters
        ----------
        edges : array-like of shape (num_edges, 2)
            An array of edge pairs, where each pair (i, j) represents an edge between vertices i and j.
        pos : array-like of shape (num_vertices, 2)
            A 2D array containing the (x, y) coordinates of each vertex.

        Returns
        -------
        None
            Displays a Plotly 2D figure with vertices plotted as red markers and edges as gray lines.
        """
        pos = np.array(self.positions)
        edges = np.array(self.edges)

        x_edges = []
        y_edges = []
        for i, j in edges:
            x_edges += [pos[i, 0], pos[j, 0], None]
            y_edges += [pos[i, 1], pos[j, 1], None]

        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line=dict(color='gray', width=edge_width),
            hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode='markers',
            marker=dict(
                color=node_colors if node_colors is not None else 'red',
                colorscale='Bluered',
                size=node_size,
                colorbar=dict(title='Degree'),
                showscale=True
            ),
            hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="2D Graph Embedding",
            xaxis=dict(title="X", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
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
        edges : array-like of shape (num_edges, 2)
            An array of edge pairs, where each pair (i, j) represents an edge between vertices i and j.
        pos : array-like of shape (num_vertices, 3)
            A 3D array containing the (x, y, z) coordinates of each vertex.

        Returns
        -------
        None
            Displays a Plotly 3D figure with vertices plotted as red markers and edges as gray lines.
        """
        pos = np.array(self.positions)
        edges = np.array(self.edges)


        x_edges, y_edges, z_edges = [], [], []
        for i, j in edges:
            x_edges += [pos[i, 0], pos[j, 0], None]
            y_edges += [pos[i, 1], pos[j, 1], None]
            z_edges += [pos[i, 2], pos[j, 2], None]

        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line=dict(color='gray', width=edge_width),
            hoverinfo='none'
        )

        node_trace = go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker=dict(
                color=node_colors if node_colors is not None else 'red',
                colorscale='Bluered',
                size=node_size,
                colorbar=dict(title='Degree'),
                showscale=True
            ),
            hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Graph Embedding",
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z")
            ),
            showlegend=False,
            width=800,
            height=800
        )
        fig.show()


# In[4]:


import plotly.io as pio
# Plotly settings: either interactive images or stills
pio.renderers.default = 'colab'  # interactive plots
#pio.renderers.default = 'png'      # stills


# In[7]:


def erdos_renyi_graph(n, p, seed=0):
    """
    Generate a random undirected graph using the Erdős–Rényi G(n, p) model.

    Parameters:
      n: int
         Number of vertices.
      p: float
         Probability that an edge exists between any pair of vertices.
      seed: int
         Random seed for reproducibility.

    Returns:
      edges: np.ndarray of shape (num_edges, 2)
         Array of edge pairs (i, j) with i < j.
    """

    rng = np.random.default_rng(seed)
    # Create a random matrix for the upper triangular part
    # Note: Only the upper triangular portion (excluding the diagonal) is needed
    upper = rng.random((n, n))
    # Create a mask for where edges exist (with probability p)
    mask = (upper < p)
    # Ensure no self-loops and consider only the upper triangular part
    mask = np.triu(mask, k=1)
    # Get the indices of the true entries, i.e., the edges
    i, j = np.nonzero(mask)
    edges = np.column_stack((i, j))

    return edges

def compute_vertex_degrees(n, edges):
    """
    Compute the degree of each vertex from the edge list.

    Parameters:
      n: number of vertices
      edges: array of shape (num_edges, 2)

    Returns:
      degrees: np.array of shape (n,) with degree of each vertex
    """
    degrees = np.zeros(n, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    return degrees


def erdos_renyi_graph_test(n = 1000,
                           p = 0.025,
                           dim = 3,
                           num_iterations=20,
                           L_min=10.0,
                           k_attr=0.5,
                           k_inter=0.1,
                           knn_k=15,
                           edge_width=0.25,
                           node_size=8):
    """
    Test the layout on Erdős–Rényi graphs.
    """

    # Convert to a JAX array.
    edges = jnp.array(erdos_renyi_graph(n, p))
    logger.debug(f"Vertices {n}, edges {edges.shape[0]}")

    gm = GraphEmbedder(edges, n,
                       dimension=dim,
                       L_min=L_min,
                       k_attr=k_attr,
                       k_inter=k_inter,
                       knn_k=knn_k,
                       sample_size=1024,
                       batch_size=1024,
                       my_logger=logger)

    _ = gm.run_layout(num_iterations)

    deg = compute_vertex_degrees(n, np.array(edges))
    deg_normalized = (deg - np.min(deg)) / (np.max(deg) - np.min(deg))

    gm.display_layout(edge_width=edge_width, node_size=node_size, node_colors=deg_normalized)

# -------------------------------


# In[8]:


"""
Run the test with Erdős–Rényi graphs, and visualize the embedding.
The vertex colors are the vertex degrees (normalized to [0, 1]).
Blue = low degree, red = high degree.
"""
erdos_renyi_graph_test(n=1000)


# # Testing on various graphs and their centrality measures

# In[9]:


import networkx as nx

# Generate SBM graph with known communities
def generate_sbm(n_per_block=75, num_blocks=4, p_in=0.15, p_out=0.01, labels=False, seed=0):
    n_per_block, num_blocks, p_in, p_out, seed = int(n_per_block), int(num_blocks), float(p_in), float(p_out), int(seed)
    np.random.seed(seed)
    sizes = [n_per_block] * num_blocks
    probs = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    edges = np.array(G.edges())
    true_labels = []
    for i, size in enumerate(sizes):
        true_labels += [i] * size
    if labels:
        return edges, G, true_labels
    return edges, G

# Barabási–Albert model: scale-free network
def generate_ba(n=300, m=3, seed=0):
    n, m, seed = int(n), int(m), int(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    edges = np.array(G.edges())
    return edges, G

# Watts-Strogaz model
def generate_ws(n=1000, k=6, p=0.3, seed=0):
    n, k, p, seed = int(n), int(k), float(p), int(seed)
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    edges = np.array(G.edges())
    return edges, G

# Power cluster model
def generate_power_cluster(n=1000, m=3, p=0.5, seed=0):
    n, m, p, seed = int(n), int(m), float(p), int(seed)
    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    edges = np.array(G.edges())
    return edges, G

# Road network: 2D grid graph
def generate_road_network(width=30, height=30):
    width, height = int(width), int(height)
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(width, height))
    edges = np.array(G.edges())
    return edges, G

# Bipartite graph
def generate_bipartite_graph(n_top=50, n_bottom=100):
    n_top, n_bottom = int(n_top), int(n_bottom)
    G = nx.complete_bipartite_graph(n_top, n_bottom)
    edges = np.array(G.edges())
    return edges, G

# Balanced balanced tree
def generate_balanced_tree(r=2, h=10):
    r, h = int(r), int(h)
    G = nx.balanced_tree(r, h)
    edges = np.array(G.edges())
    return edges, G


# In[11]:


"""

Test the layout on graphs generated above.

"""
def graph_test(graph_generator,
               graph_params,
               dim = 3,
               num_iterations=40,
               L_min=10.0,
               k_attr=0.5,
               k_inter=0.1,
               knn_k=15,
               edge_width=0.5,
               node_size=8,
               sample_size=512,
               batch_size=1024):

    # Convert to a JAX array.
    edges, G = graph_generator(**graph_params)
    edges = jnp.array(edges)
    n = G.number_of_nodes()
    deg = compute_vertex_degrees(n, np.array(edges))
    deg_normalized = (deg - np.min(deg)) / (np.max(deg) - np.min(deg))

    logger.debug(f"Vertices {n}, edges {edges.shape[0]}")

    gm = GraphEmbedder(edges, n,
                       dimension=dim,
                       L_min=L_min,
                       k_attr=k_attr,
                       k_inter=k_inter,
                       knn_k=knn_k,
                       sample_size=sample_size,
                       batch_size=batch_size,
                       my_logger=logger)

    gm.display_layout(edge_width=edge_width, node_size=node_size, node_colors=deg_normalized)
    _ = gm.run_layout(num_iterations)
    gm.display_layout(edge_width=edge_width, node_size=node_size, node_colors=deg_normalized)

# -------------------------------


# In[12]:


graph_test(generate_bipartite_graph, {"n_top": 50, "n_bottom": 100}, dim=2, L_min=20, num_iterations=5)


# In[13]:


graph_test(generate_balanced_tree, {"r": 3, "h": 8}, dim=2, sample_size=2048, batch_size=512, num_iterations=60)


# In[11]:


"""
Grid (road network) graph
"""
graph_test(generate_road_network, {"width": 30, "height": 40}, dim=2, num_iterations=60)
#


# In[12]:


"""
Test power cluster graph
"""
graph_test(generate_power_cluster, {"n": 1000, "m": 5, "p": 0.75}, dim=2, sample_size=4096, batch_size=512, num_iterations=60)


# In[13]:


"""
Test Barabási–Albert graph
"""
graph_test(generate_ba, {"n": 1000, "m": 5}, dim=2, L_min=20, sample_size=2048, batch_size=512, num_iterations=60)


# In[14]:


"""
Test SBM graph
"""
graph_test(generate_sbm, {"n_per_block": 100, "num_blocks": 4, "p_in": 0.15, "p_out": 0.01}, dim=2, L_min=60, sample_size=1024, batch_size=512, num_iterations=80)


# In[15]:


graph_test(generate_ws, {"n": 1000, "k": 6, "p": 0.3}, dim=2, L_min=40, sample_size=2048, batch_size=512, num_iterations=60)


# In[16]:


from scipy.stats import pearsonr, spearmanr
import pandas as pd

# --- Correlation (Pearson's r, Spearman's rho) ---
def report_corr(name, radii, centrality, alpha=0.025):
    q_low, q_high = np.quantile(radii, [alpha, 1-alpha])
    mask = (radii >= q_low) & (radii <= q_high)
    pr, p_val_pr = pearsonr(radii[mask], centrality[mask])
    print(f"{name:25s} ↔ radial distance: Pearson's r = {pr:.3f}, p = {p_val_pr:.2e}")
    sr, p_val_sr = spearmanr(radii, centrality)
    print(f"{name:25s} ↔ radial distance: Spearman's rho = {sr:.3f}, p = {p_val_sr:.2e}")

# --- Full correlation matrix ---
def report_full_correlation_matrix(radii, deg, btw, eig, pr, clo, edge_btw, alpha=0.025):
    names = [
        "Radial Distance",
        "Degree Centrality",
        "Betweenness",
        "Eigenvector",
        "PageRank",
        "Closeness",
        "Edge Betweenness"
    ]
    vectors = [radii, deg, btw, eig, pr, clo, edge_btw]

    q_low, q_high = np.quantile(radii, [alpha, 1-alpha])
    mask = (radii >= q_low) & (radii <= q_high)

    n = len(names)
    pearson_mat = np.zeros((n, n))
    spearman_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pearson_mat[i, j], _ = pearsonr(vectors[i][mask], vectors[j][mask])
            spearman_mat[i, j], _ = spearmanr(vectors[i][mask], vectors[j][mask])

    # Convert to DataFrames for pretty printing or export
    pearson_df = pd.DataFrame(pearson_mat, index=names, columns=names)
    spearman_df = pd.DataFrame(spearman_mat, index=names, columns=names)

    print("\n Pearson Correlation Matrix:")
    print(pearson_df.round(3))

    print("\n Spearman Correlation Matrix:")
    print(spearman_df.round(3))

    return pearson_df, spearman_df


# In[17]:


import math
import matplotlib.pyplot as plt

def plot_radial_vs_centrality(radii, centralities, names, alpha=0.025):
    num_plots = len(centralities)
    cols = 3
    rows = math.ceil(num_plots / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (c, name) in enumerate(zip(centralities, names)):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(c, radii, alpha=0.5, s=10)
        plt.xlabel(name)
        plt.ylabel("Radial distance")
        q_low, q_high = np.quantile(radii, [alpha, 1-alpha])
        mask = (radii >= q_low) & (radii <= q_high)
        rp, _ = pearsonr(radii[mask], c[mask])
        rs, _ = spearmanr(radii[mask], c[mask])

        plt.title(f"{name}\nPearson r = {rp:.3f}, \nSpearman rho = {rs:.3f}")

    plt.tight_layout()
    plt.show()


# In[18]:


"""
Running and displaying a benchmark
"""

def run_benchmark(graph_generator,
                  graph_params,
                  dim = 3,
                  L_min=10.0,
                  k_attr=0.5,
                  k_inter=0.1,
                  knn_k=15,
                  edge_width=0.5,
                  node_size=8,
                  sample_size=512,
                  batch_size=1024,
                  num_iterations=40):

    # Convert to a JAX array.
    edges, G = graph_generator(**graph_params)
    edges = jnp.array(edges)
    n = G.number_of_nodes()
    deg = compute_vertex_degrees(n, np.array(edges))
    deg_normalized = (deg - np.min(deg)) / (np.max(deg) - np.min(deg))

    logger.debug(f"Vertices {n}, edges {edges.shape[0]}")

    gm = GraphEmbedder(edges, n,
                       dimension=dim,
                       L_min=L_min,
                       k_attr=k_attr,
                       k_inter=k_inter,
                       knn_k=knn_k,
                       sample_size=sample_size,
                       batch_size=batch_size,
                       my_logger=logger)

    # Radial distance
    positions = gm.run_layout(num_iterations)
    radii = np.linalg.norm(positions, axis=1)

    # Centralities
    deg_centrality = np.array(list(nx.degree_centrality(G).values()))
    btw_centrality = np.array(list(nx.betweenness_centrality(G).values()))
    eig_centrality = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    pagerank = np.array(list(nx.pagerank(G).values()))
    closeness = np.array(list(nx.closeness_centrality(G).values()))

    # Convert edge betweenness into node-level centrality
    edge_betweenness_centrality = nx.edge_betweenness_centrality(G)
    edge_btw_node = np.zeros(n)

    for (u, v), val in edge_betweenness_centrality.items():
      edge_btw_node[u] += val
      edge_btw_node[v] += val  # count both ends

    # Correlation printout
    report_corr("Degree centrality", radii, deg_centrality)
    report_corr("Betweenness centrality", radii, btw_centrality)
    report_corr("Eigenvector centrality", radii, eig_centrality)
    report_corr("PageRank", radii, pagerank)
    report_corr("Closeness centrality", radii, closeness)
    report_corr("Edge Betweenness centrality", radii, edge_btw_node)

    # Full correlation matrix
    report_full_correlation_matrix(radii, deg_centrality, btw_centrality, eig_centrality, pagerank, closeness, edge_btw_node)

    # Plot
    plot_radial_vs_centrality(
      radii,
      [deg_centrality, btw_centrality, eig_centrality, pagerank, closeness, edge_btw_node],
      ["Degree Centrality", "Betweenness", "Eigenvector", "Pagerank", "Closeness Centrality", "Edge Betweenness Centrality"]
    )


# In[19]:


"""
SBM graph: community structure
"""
run_benchmark(generate_sbm, {"n_per_block": 100, "num_blocks": 4, "p_in": 0.15, "p_out": 0.01}, dim=2, L_min=60, sample_size=1024, batch_size=512, num_iterations=80)


# In[20]:


"""
Barabási–Albert model: scale-free network
"""

run_benchmark(generate_ba, {"n": 1000, "m": 5}, dim=2, L_min=20, sample_size=2048, batch_size=512, num_iterations=60)


# In[21]:


"""
Watts-Strogaz model
"""

run_benchmark(generate_ws, {"n": 1000, "k": 6, "p": 0.3}, dim=2, L_min=40, sample_size=2048, batch_size=512, num_iterations=60)


# In[22]:


"""
Power cluster model
"""

run_benchmark(generate_power_cluster, {"n": 1000, "m": 5, "p": 0.5}, sample_size=2048, batch_size=512, num_iterations=60)


# In[23]:


"""
Road network: 2D grid graph
"""

run_benchmark(generate_road_network, {"width": 30, "height": 40}, dim=2, num_iterations=60)


# In[24]:


"""
Bipartite graph
"""

run_benchmark(generate_bipartite_graph, {"n_top": 50, "n_bottom": 100}, dim=2, num_iterations=5)


# In[25]:


"""
Balanced binary tree
"""

run_benchmark(generate_balanced_tree, {"r": 2, "h": 10}, dim=2, sample_size=2046, batch_size=512, num_iterations=60)


# In[26]:


test_graphs = [

      (generate_sbm, "SBM Graph",
      {"n_per_block": 100, "num_blocks": 4, "p_in": 0.15, "p_out": 0.01},
      {"dim": 2, "L_min": 60, "sample_size": 1024, "batch_size": 512, "num_iterations": 80}),

      (generate_ba, "Barabási–Albert Graph",
      {"n": 1000, "m": 5},
      {"dim": 2, "L_min": 20, "sample_size": 2048, "batch_size": 512, "num_iterations": 60}),

      (generate_ws, "Watts-Strogaz Graph",
      {"n": 1000, "k": 6, "p": 0.3},
      {"dim": 2, "L_min": 40, "sample_size": 2048, "batch_size": 512, "num_iterations": 60}),

      (generate_power_cluster, "Powerlaw Cluster Graph",
      {"n": 1000, "m": 5, "p": 0.5},
      {"sample_size": 2048, "batch_size": 512, "num_iterations": 60}),

      (generate_road_network, "Road Network (Grid)",
      {"width": 30, "height": 40},
      {"dim": 2, "num_iterations": 60}),

      (generate_bipartite_graph, "Bipartite Graph",
      {"n_top": 50, "n_bottom": 100},
      {"dim": 2, "num_iterations": 5}),

      (generate_balanced_tree, "Balanced Binary Tree",
      {"r": 2, "h": 10},
      {"dim": 2, "sample_size": 2046, "batch_size": 512, "num_iterations": 60})

  ]


# In[27]:


def benchmark_correlations(graph_generator, graph_params, dim=2, L_min=10.0, k_attr=0.5, k_inter=0.1, knn_k=15, sample_size=512, batch_size=1024, num_iterations=40):

    edges, G = graph_generator(**graph_params)
    edges = jnp.array(edges)
    n = G.number_of_nodes()

    gm = GraphEmbedder(edges, n,
                      dimension=dim,
                      L_min=L_min,
                      k_attr=k_attr,
                      k_inter=k_inter,
                      knn_k=knn_k,
                      sample_size=sample_size,
                      batch_size=batch_size,
                      my_logger=None)
    positions = gm.run_layout(num_iterations)
    positions = np.array(positions)
    radii = np.linalg.norm(positions, axis=1)


    # Classical centrality measures
    deg = np.array(list(nx.degree_centrality(G).values()))
    btw = np.array(list(nx.betweenness_centrality(G).values()))
    eig = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    pr = np.array(list(nx.pagerank(G).values()))
    clo = np.array(list(nx.closeness_centrality(G).values()))

    # Edge betweenness aggregated to nodes
    edge_btw = nx.edge_betweenness_centrality(G)
    edge_btw_node = np.zeros(n)
    for (u, v), val in edge_btw.items():
        edge_btw_node[u] += val
        edge_btw_node[v] += val

    metrics = {
        "Degree": deg,
        "Betweenness": btw,
        "Eigenvector": eig,
        "PageRank": pr,
        "Closeness": clo,
        "Edge Betweenness": edge_btw_node
    }

    correlations = {}
    for label, vec in metrics.items():
        pear_r, _ = pearsonr(radii, vec)
        spear_r, _ = spearmanr(radii, vec)
        correlations[label] = {"Pearson": float(pear_r), "Spearman": float(spear_r)}

    return correlations


# In[28]:


benchmark_results = {}
for graph_generator, graph_name, graph_params, params in tqdm(test_graphs):
    logger.info(f"Running benchmark for {graph_name}")
    benchmark_results[graph_name] = benchmark_correlations(graph_generator, graph_params, **params)


# In[29]:


def display_benchmark_results(benchmark_results):
  """Displays benchmark results in a nicely formatted table."""

  data = []
  for graph_name, correlations in benchmark_results.items():
    row = {"Graph Name": graph_name}
    for metric, corr_values in correlations.items():
      row[f"{metric} (Pearson)"] = corr_values["Pearson"]
      row[f"{metric} (Spearman)"] = corr_values["Spearman"]
    data.append(row)

  df = pd.DataFrame(data).set_index("Graph Name")
  print(df.round(3))

display_benchmark_results(benchmark_results)


# In[34]:


import time
import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

# =============================================================================
# Influence Maximization via Embedding-Based Seed Selection
# =============================================================================

def graphem_seed_selection(embedder, k, num_iterations=20):
    """
    Run the GraphEmbedder layout to get an embedding, then select
    k seeds by choosing the nodes with the highest radial distances.
    """
    final_positions = embedder.run_layout(num_iterations=num_iterations)
    positions = np.array(final_positions)
    radial = np.linalg.norm(positions, axis=1)
    sorted_indices = np.argsort(-radial)
    seed_indices = sorted_indices[:k]
    return seed_indices, radial

# =============================================================================
# NDlib-based Influence Estimation
# =============================================================================

def ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=200):
    """
    Run NDlib's Independent Cascades model on graph G, starting with the given seeds,
    and return the estimated final influence (number of nodes in state 2) and
    the number of iterations executed.
    """
    model = ep.IndependentCascadesModel(G)
    config = mc.Configuration()
    config.add_model_parameter('fraction_infected', 0.1)
    for e in G.edges():
        config.add_edge_configuration("threshold", e, p)
    model.set_initial_status(config)
    sim_iterations = model.iteration_bunch(iterations_count)
    final_count = sim_iterations[-1]['node_count']
    influence = final_count.get(2, 0)
    return influence, len(sim_iterations)

# =============================================================================
# Greedy Seed Selection for Influence Maximization
# =============================================================================

def greedy_seed_selection(G, k, p=0.1, iterations_count=200):
    """
    Greedy seed selection using NDlib influence estimation.
    For each candidate node evaluation, it calls NDlib's simulation and accumulates
    the total number of iterations used across all evaluations.

    Returns:
        seeds: the selected seed set (list of nodes)
        total_iters: the total number of NDlib iterations run during selection.
    """
    seeds = []
    candidate_nodes = set(G.nodes())
    total_iters = 0
    for _ in range(k):
        best_candidate = None
        best_spread = -1
        # Evaluate each candidate's marginal gain when added to the current seed set.
        for node in candidate_nodes:
            current_seeds = seeds + [node]
            spread, iters = ndlib_estimated_influence(G, current_seeds, p=p, iterations_count=iterations_count)
            total_iters += iters  # accumulate iterations used for this simulation
            if spread > best_spread:
                best_spread = spread
                best_candidate = node
        seeds.append(best_candidate)
        candidate_nodes.remove(best_candidate)
    return seeds, total_iters

# =============================================================================
# Main Benchmarking Routine
# =============================================================================

n_nodes = 128
p_edge = 0.05
ic_prob = 0.1
k_seeds = 10

def run_benchmark():

  # Create a sample graph (Erdős–Rényi)
  G_nx = nx.erdos_renyi_graph(n_nodes, p_edge, seed=42)
  edges = np.array(G_nx.edges())
  edges = np.sort(edges, axis=1)

  # -------------------------------
  # HPIndex-based Embedding Method
  # -------------------------------
  embedder = GraphEmbedder(edges=edges, n_vertices=n_nodes, dimension=2,
                          L_min=10.0, k_attr=0.5, k_inter=0.1, knn_k=15,
                          sample_size=256, batch_size=1024, verbose=False)

  start_time = time.time()
  gm_seeds, radial = graphem_seed_selection(embedder, k_seeds, num_iterations=20)
  gm_time = time.time() - start_time
  gm_influence, gm_iter_count = ndlib_estimated_influence(G_nx, gm_seeds, p=ic_prob, iterations_count=200)

  # -------------------------------------
  # Greedy Influence Maximization Method
  # -------------------------------------
  start_time = time.time()
  greedy_seeds, greedy_iters = greedy_seed_selection(G_nx, k_seeds, p=ic_prob, iterations_count=200)
  greedy_time = time.time() - start_time
  greedy_influence, iters = ndlib_estimated_influence(G_nx, greedy_seeds, p=ic_prob, iterations_count=200)
  greedy_iters += iters  # accumulate iterations used for the final simulation

  return gm_seeds, gm_influence, gm_iter_count, gm_time, greedy_seeds, greedy_influence, greedy_iters, greedy_time

N = 10 # Sample size

gm_seeds_stats = []
gm_influence_stats = []
gm_iter_count_stats = []
gm_time_stats = []
greedy_seeds_stats = []
greedy_influence_stats = []
greedy_iters_stats = []
greedy_time_stats = []

for _ in range(N):

  print("Iteration", _+1, "of", N)

  gm_seeds, gm_influence, gm_iter_count, gm_time, greedy_seeds, greedy_influence, greedy_iters, greedy_time = run_benchmark()

  gm_seeds_stats.append(gm_seeds)
  gm_influence_stats.append(gm_influence)
  gm_iter_count_stats.append(gm_iter_count)
  gm_time_stats.append(gm_time)

  greedy_seeds_stats.append(greedy_seeds)
  greedy_influence_stats.append(greedy_influence)
  greedy_iters_stats.append(greedy_iters)
  greedy_time_stats.append(greedy_time)

gm_seeds_stats = np.array(gm_seeds_stats)
gm_influence_stats = np.array(gm_influence_stats)
gm_iter_count_stats = np.array(gm_iter_count_stats)
gm_time_stats = np.array(gm_time_stats)

greedy_seeds_stats = np.array(greedy_seeds_stats)
greedy_influence_stats = np.array(greedy_influence_stats)
greedy_iters_stats = np.array(greedy_iters_stats)
greedy_time_stats = np.array(greedy_time_stats)

print(" \nGraphEm Embedding Method:")
print("  Estimated Influence Spread:", gm_influence_stats.mean(), "(sigma)", gm_influence_stats.std())
print("  NDlib Iterations:", gm_iter_count_stats.mean(), "(sigma)", gm_iter_count_stats.std())
print("  Runtime (s):", gm_time_stats.mean(), "(sigma)", gm_time_stats.std())

print("\nGreedy Influence Maximization Method:")
print("  Estimated Influence Spread:", greedy_influence_stats.mean(), "(sigma)", greedy_influence_stats.std())
print("  NDlib Iterations:", greedy_iters_stats.mean(), "(sigma)", greedy_iters_stats.std())
print("  Runtime (s):", greedy_time_stats.mean(), "(sigma)", greedy_time_stats.std())


# In[30]:




