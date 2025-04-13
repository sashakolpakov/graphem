"""
Graph Embedding functionality for Graphem.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from loguru import logger

from graphem.index import HPIndex

"""
Optimized Graph Embedding functionality for Graphem.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from loguru import logger
from functools import partial

from graphem.index import HPIndex


class GraphEmbedder:
    """
    A class for embedding graphs using the Laplacian embedding with JAX optimizations.

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

    def __init__(self, edges, n_vertices, dimension=2, L_min=1.0, k_attr=0.2, k_inter=0.5, 
                 knn_k=10, sample_size=256, batch_size=1024, my_logger=None, verbose=True):
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
        
        # Initial positions
        self.positions = self._laplacian_embedding()
        
        # JIT compile the spring forces computation
        self.compute_spring_forces_jit = jax.jit(
            lambda positions: self._compute_spring_forces(
                positions, self.edges, self.L_min, self.k_attr
            )
        )
        
        # Pre-compile vectorized repulsion calculation
        self.repulse_vectorized_jit = jax.jit(self._repulse_vectorized)

    def _laplacian_embedding(self):
        """
        Compute the Laplacian embedding of the graph.
        """
        # Create adjacency matrix
        adjacency = np.zeros((self.n, self.n))
        for i, j in self.edges:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

        # Compute degree matrix and Laplacian
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency

        # Compute eigenvectors and eigenvalues of Laplacian
        eigval, eigvec = np.linalg.eigh(laplacian)

        # Use first non-zero eigenvectors as embedding
        # (skip 0th eigenvector which corresponds to the constant vector)
        dim = min(self.dimension, self.n - 1)
        
        # Create the initial embedding
        embedding = eigvec[:, 1:dim+1]
        
        # Scale by 10 to get a reasonable initial size
        embedding = embedding * 10

        return jnp.array(embedding)

    # JIT-compiled function to locate KNN midpoints
    @partial(jax.jit, static_argnums=(2,))
    def _locate_knn_midpoints_internal(self, midpoints, k):
        """JIT-compiled internal function for KNN calculation."""
        # This is simplified for demonstration; in practice, you might need
        # to adapt the HPIndex.knn_tiled function for JIT compilation
        distances = jax.vmap(
            lambda x: jnp.sqrt(jnp.sum((midpoints - x)**2, axis=1))
        )(midpoints)
        
        # Get indices of k+1 smallest distances (including self)
        indices = jnp.argsort(distances, axis=1)[:, :k+1]
        
        # Remove self-neighbor (first column)
        return indices[:, 1:]
    
    def locate_knn_midpoints(self, midpoints, k):
        """
        Locate k nearest neighbors for each midpoint.
        """
        # Use our efficient KNN search
        # For now, we'll use the existing implementation since HPIndex may not be JIT-compatible
        knn_idx = HPIndex.knn_tiled(
            midpoints, midpoints, k=k+1, 
            x_tile_size=min(midpoints.shape[0], 8192), 
            y_batch_size=min(midpoints.shape[0], 1024)
        )
        
        # Remove self-neighbor (i.e., the first index which is the point itself)
        return knn_idx[:, 1:]

    @staticmethod
    def _compute_spring_forces(positions, edges, L_min, k_attr):
        """
        Compute the spring forces between vertices using vectorized operations.
        """
        # Extract edge endpoints
        u = edges[:, 0]
        v = edges[:, 1]
        
        # Compute edge vectors
        edge_vecs = positions[v] - positions[u]
        
        # Compute lengths
        lengths = jnp.linalg.norm(edge_vecs, axis=1)
        
        # Compute force directions (unit vectors along edges)
        directions = edge_vecs / jnp.expand_dims(jnp.maximum(lengths, 1e-8), axis=1)
        
        # Compute spring force magnitudes
        force_mags = k_attr * (lengths - L_min)
        
        # Compute forces
        forces = jnp.expand_dims(force_mags, axis=1) * directions
        
        # Initialize displacement array
        displacement = jnp.zeros_like(positions)
        
        # Update displacements using scatter-add operations
        displacement = displacement.at[u].add(forces)
        displacement = displacement.at[v].add(-forces)
        
        return displacement

    @staticmethod
    def _repulse_vectorized(vertices, midpoints):
        """Vectorized version of repulse function."""
        # Compute pairwise differences
        diff = jnp.expand_dims(vertices, 1) - jnp.expand_dims(midpoints, 0)
        
        # Compute distances
        distances = jnp.linalg.norm(diff, axis=2)
        
        # Avoid division by zero with a small epsilon
        safe_distances = jnp.maximum(distances, 1e-8)
        
        # Compute directions (unit vectors)
        directions = diff / jnp.expand_dims(safe_distances, 2)
        
        # Compute force magnitudes (inverse square law)
        magnitudes = 1.0 / (jnp.square(safe_distances) + 1e-8)
        
        # Apply magnitudes to directions
        forces = jnp.expand_dims(magnitudes, 2) * directions
        
        return forces
    
    # JIT-compiled intersection forces computation
    @partial(jax.jit, static_argnums=(0,))
    def _compute_intersection_forces_jit(self, positions, edges, knn_indices, sampled_indices, k_inter):
        """JIT-compiled function to compute intersection forces."""
        # Compute midpoints of all edges
        midpoints = (positions[edges[:, 0]] + positions[edges[:, 1]]) / 2
        
        # Get sampled midpoints and their edges
        sampled_midpoints = midpoints[sampled_indices]
        sampled_edges = edges[sampled_indices]
        
        # Get neighbor midpoints and edges
        neighbor_indices = knn_indices.reshape(-1)
        neighbor_midpoints = midpoints[neighbor_indices].reshape(sampled_indices.shape[0], -1, positions.shape[1])
        neighbor_edges = edges[neighbor_indices].reshape(sampled_indices.shape[0], -1, 2)
        
        # Create vectors for vertices
        # For each sampled edge
        u_sampled = positions[sampled_edges[:, 0]]
        v_sampled = positions[sampled_edges[:, 1]]
        
        # For each neighbor edge (reshape for broadcasting)
        u_neighbor = jnp.take(positions, neighbor_edges[..., 0], axis=0)
        v_neighbor = jnp.take(positions, neighbor_edges[..., 1], axis=0)
        
        # Check adjacency - vertices should not be shared
        u_sampled_expanded = jnp.expand_dims(sampled_edges[:, 0], 1)
        v_sampled_expanded = jnp.expand_dims(sampled_edges[:, 1], 1)
        not_adjacent = (neighbor_edges[..., 0] != u_sampled_expanded) & \
                       (neighbor_edges[..., 0] != v_sampled_expanded) & \
                       (neighbor_edges[..., 1] != u_sampled_expanded) & \
                       (neighbor_edges[..., 1] != v_sampled_expanded)
        
        # Calculate repulsion forces using vectorized function
        # Repulsion from neighbor midpoints to sampled vertices
        repulsion_u_to_neighbor = jax.vmap(lambda u, mids: jax.vmap(
            lambda mid: GraphEmbedder._repulse(u, mid)
        )(mids))(u_sampled, neighbor_midpoints)
        
        repulsion_v_to_neighbor = jax.vmap(lambda v, mids: jax.vmap(
            lambda mid: GraphEmbedder._repulse(v, mid)
        )(mids))(v_sampled, neighbor_midpoints)
        
        # Apply adjacency mask
        not_adjacent_expanded = jnp.expand_dims(not_adjacent, -1)
        repulsion_u_to_neighbor = repulsion_u_to_neighbor * not_adjacent_expanded
        repulsion_v_to_neighbor = repulsion_v_to_neighbor * not_adjacent_expanded
        
        # Scale by k_inter
        repulsion_u_to_neighbor = k_inter * repulsion_u_to_neighbor
        repulsion_v_to_neighbor = k_inter * repulsion_v_to_neighbor
        
        # Initialize displacement array
        displacement = jnp.zeros_like(positions)
        
        # First add forces to sampled vertices
        for i in range(sampled_indices.shape[0]):
            # Add force to u vertex
            displacement = displacement.at[sampled_edges[i, 0]].add(
                jnp.sum(repulsion_u_to_neighbor[i], axis=0))
            
            # Add force to v vertex
            displacement = displacement.at[sampled_edges[i, 1]].add(
                jnp.sum(repulsion_v_to_neighbor[i], axis=0))
        
        return displacement

    @staticmethod
    def _repulse(vertex, mid):
        """Compute repulsion force vector."""
        diff = vertex - mid
        distance = jnp.linalg.norm(diff)
        # Avoid division by very small numbers
        direction = diff / (distance + 1e-8)
        # Inverse square law for repulsion
        magnitude = 1.0 / (distance * distance + 1e-8)
        return magnitude * direction

    # JIT-compiled position update step
    @partial(jax.jit, static_argnums=(0,))
    def _update_positions_step_jit(self, positions, edges, sampled_indices, knn_indices):
        """JIT-compiled function to perform one step of position updates."""
        # Compute spring forces using the jitted function
        spring_forces = self._compute_spring_forces(positions, edges, self.L_min, self.k_attr)
        
        # Compute intersection forces using the jitted function
        intersection_forces = self._compute_intersection_forces_jit(
            positions, edges, knn_indices, sampled_indices, self.k_inter
        )
        
        # Combine forces
        total_forces = spring_forces + intersection_forces
        
        # Update positions
        new_positions = positions + 0.1 * total_forces
        
        # Apply clipping for 3D case
        if positions.shape[1] == 3:
            new_positions = new_positions.at[:, 2].set(
                jnp.clip(new_positions[:, 2], -5.0, 5.0)
            )
        
        return new_positions

    def update_positions(self):
        """
        Update the positions of the vertices based on the spring forces and intersection forces.
        """
        # Sample midpoints for intersection detection
        self.logger.info(f"Sampling {self.sample_size} midpoints for intersection detection")
        n_edges = self.edges.shape[0]
        sampled_indices = jnp.array(np.random.choice(
            n_edges, size=min(self.sample_size, n_edges), replace=False
        ))
        
        # Compute midpoints of edges
        midpoints = (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]) / 2
        
        # Find k nearest neighbors for each midpoint
        knn_idx = self.locate_knn_midpoints(
            midpoints[sampled_indices], k=self.knn_k
        )
        
        # Use the JIT-compiled function to update positions
        self.positions = self._update_positions_step_jit(
            self.positions, self.edges, sampled_indices, knn_idx
        )

    def run_layout(self, num_iterations=100):
        """
        Run the layout for a given number of iterations.
        """
        self.logger.info(f"Running layout for {num_iterations} iterations")
        
        for i in range(num_iterations):
            self.update_positions()
            if (i + 1) % 10 == 0:
                self.logger.info(f"Iteration {i+1}/{num_iterations}")

    def display_layout(self, edge_width=1, node_size=3, node_colors=None):
        """
        Display the graph embedding using Plotly.
        """
        if self.dimension == 2:
            self._display_layout_2d(edge_width, node_size, node_colors)
        elif self.dimension == 3:
            self._display_layout_3d(edge_width, node_size, node_colors)
        else:
            raise ValueError(f"Cannot display {self.dimension}D layout")

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
        # Convert positions to numpy for easier handling
        positions = np.array(self.positions)

        # Create an empty figure
        fig = go.Figure()

        # Add edges as scatter line objects
        for i, j in self.edges:
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], positions[j, 0]],
                y=[positions[i, 1], positions[j, 1]],
                mode='lines',
                line=dict(width=edge_width, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

        # Add vertices as scatter points
        x, y = positions[:, 0], positions[:, 1]
        
        # Default color to red if not specified
        if node_colors is None:
            node_colors = 'red'
            
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                line=dict(width=0)
            ),
            hovertext=[f"Vertex {i}" for i in range(len(x))],
            hoverinfo='text',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title="2D Graph Embedding",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor='white'
        )

        # Display the figure
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
        # Convert positions to numpy for easier handling
        positions = np.array(self.positions)

        # Create an empty figure
        fig = go.Figure()

        # Add edges as scatter line objects
        for i, j in self.edges:
            fig.add_trace(go.Scatter3d(
                x=[positions[i, 0], positions[j, 0]],
                y=[positions[i, 1], positions[j, 1]],
                z=[positions[i, 2], positions[j, 2]],
                mode='lines',
                line=dict(width=edge_width, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

        # Add vertices as scatter points
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Default color to red if not specified
        if node_colors is None:
            node_colors = 'red'
            
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                line=dict(width=0)
            ),
            hovertext=[f"Vertex {i}" for i in range(len(x))],
            hoverinfo='text',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title="3D Graph Embedding",
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                )
            )
        )

        # Display the figure
        fig.show()
