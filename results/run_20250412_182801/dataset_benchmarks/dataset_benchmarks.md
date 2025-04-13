# Dataset Benchmarks

## Dataset Information

The datasets used in this benchmark are from the following sources:

- **SNAP**: Stanford Network Analysis Project datasets (prefixed with 'snap-')
  - **facebook_combined**: Facebook social network
  - **ca-GrQc**: Collaboration network of Arxiv General Relativity
  - **ca-HepTh**: Collaboration network of Arxiv High Energy Physics Theory
- **Network Repository**: Various network datasets (prefixed with 'netrepo-')
- **Semantic Scholar**: Academic citation networks (prefixed with 'semanticscholar-')

## Column Descriptions

- **dataset**: Name of the dataset
- **original_vertices**: Number of vertices in the original dataset
- **original_edges**: Number of edges in the original dataset
- **sampled_vertices**: Number of vertices after sampling (if applied)
- **sampled_edges**: Number of edges after sampling (if applied)
- **density**: Edge density of the graph (2|E|/(|V|(|V|-1)))
- **avg_degree**: Average degree of vertices (2|E|/|V|)
- **lcc_size**: Size of the largest connected component
- **lcc_fraction**: Fraction of graph in the largest connected component
- **layout_time**: Time (in seconds) to compute the graph layout
- **degree_correlation**: Spearman correlation between radial distance and degree centrality
- **betweenness_correlation**: Spearman correlation between radial distance and betweenness centrality
- **eigenvector_correlation**: Spearman correlation between radial distance and eigenvector centrality
- **pagerank_correlation**: Spearman correlation between radial distance and PageRank

| dataset                |   original_vertices |   original_edges |   sampled_vertices |   sampled_edges |   density |   avg_degree |   lcc_size |   lcc_fraction | layout_time   | degree_correlation   | betweenness_correlation   | eigenvector_correlation   | pagerank_correlation   |
|:-----------------------|--------------------:|-----------------:|-------------------:|----------------:|----------:|-------------:|-----------:|---------------:|:--------------|:---------------------|:--------------------------|:--------------------------|:-----------------------|
| snap-facebook_combined |                4039 |            88234 |               2000 |           22405 |    0.0112 |       22.405 |       1506 |         0.753  | 8.54s         | N/A                  | N/A                       | N/A                       | N/A                    |
| snap-ca-GrQc           |               26197 |            14484 |               2000 |              65 |    0      |        0.065 |          9 |         0.0045 | 6.53s         | 0.782                | 0.263                     | 0.035                     | 0.782                  |
| snap-ca-HepTh          |               68746 |            25973 |               2000 |              23 |    0      |        0.023 |          4 |         0.002  | 6.55s         | 0.487                | 0.175                     | 0.026                     | 0.487                  |


*Generated on: 2025-04-12 18:29:55*