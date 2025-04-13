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

| dataset                | error                       |
|:-----------------------|:----------------------------|
| snap-facebook_combined | name 'stats' is not defined |
| snap-ca-GrQc           | name 'stats' is not defined |
| snap-ca-HepTh          | name 'stats' is not defined |


*Generated on: 2025-04-12 16:59:01*