# Influence Benchmarks

## Influence Maximization Information

This benchmark compares different seed selection strategies for influence maximization:

- **GraphEm**: Our method that selects seeds based on the graph embedding
- **Greedy**: The greedy algorithm that iteratively selects the best node
- **Random**: Randomly selected seeds as a baseline

## Column Descriptions

- **graph_type**: Type of graph generator used
- **vertices**: Number of vertices in the graph
- **edges**: Number of edges in the graph
- **avg_degree**: Average degree of vertices (2|E|/|V|)
- **graphem_influence**: Number of nodes influenced by GraphEm seeds
- **greedy_influence**: Number of nodes influenced by greedy selection seeds
- **random_influence**: Number of nodes influenced by random seeds
- **graphem_time**: Time (in seconds) to select seeds using GraphEm
- **greedy_time**: Time (in seconds) to select seeds using greedy algorithm
- **graphem_norm_influence**: GraphEm influence normalized by graph size
- **greedy_norm_influence**: Greedy influence normalized by graph size
- **random_norm_influence**: Random influence normalized by graph size
- **graphem_efficiency**: GraphEm influence per second
- **greedy_efficiency**: Greedy influence per second

| graph_type      |   vertices |   edges |   avg_degree |   graphem_influence |   greedy_influence |   random_influence | graphem_time   | greedy_time   |   graphem_norm_influence |   greedy_norm_influence |   random_norm_influence |   graphem_efficiency |   greedy_efficiency |
|:----------------|-----------:|--------:|-------------:|--------------------:|-------------------:|-------------------:|:---------------|:--------------|-------------------------:|------------------------:|------------------------:|---------------------:|--------------------:|
| Erdős–Rényi     |        200 |     977 |         9.77 |                   0 |                  0 |                  0 | 0.00s          | 7.10s         |                        0 |                       0 |                       0 |                    0 |                   0 |
| Random Regular  |        200 |     400 |         4    |                   0 |                  0 |                  0 | 0.00s          | 5.96s         |                        0 |                       0 |                       0 |                    0 |                   0 |
| Watts-Strogatz  |        200 |     400 |         4    |                   0 |                  0 |                  0 | 0.00s          | 5.85s         |                        0 |                       0 |                       0 |                    0 |                   0 |
| Barabási-Albert |        200 |     591 |         5.91 |                   0 |                  0 |                  0 | 0.00s          | 6.38s         |                        0 |                       0 |                       0 |                    0 |                   0 |
| SBM             |        200 |    1110 |        11.1  |                   0 |                  0 |                  0 | 0.00s          | 7.74s         |                        0 |                       0 |                       0 |                    0 |                   0 |


*Generated on: 2025-04-12 18:57:22*