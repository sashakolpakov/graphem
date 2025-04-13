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

| dataset                | original_vertices   | original_edges   | sampled_vertices   | sampled_edges   | density   | avg_degree   | lcc_size   | lcc_fraction   | layout_time   | degree_correlation   | betweenness_correlation   | eigenvector_correlation   | pagerank_correlation   | error                                         |
|:-----------------------|:--------------------|:-----------------|:-------------------|:----------------|:----------|:-------------|:-----------|:---------------|:--------------|:---------------------|:--------------------------|:--------------------------|:-----------------------|:----------------------------------------------|
| snap-facebook_combined | 4,039               | 88,234           | 100                | 64              | 0.0129    | 1.2800       | 9.0000     | 0.0900         | 6.35s         | 0.730                | 0.494                     | 0.211                     | 0.740                  | nan                                           |
| snap-ca-GrQc           | N/A                 | N/A              | N/A                | N/A             | N/A       | N/A          | N/A        | N/A            | N/A           | N/A                  | N/A                       | N/A                       | N/A                    | index is out of bounds for axis 0 with size 0 |
| snap-ca-HepTh          | N/A                 | N/A              | N/A                | N/A             | N/A       | N/A          | N/A        | N/A            | N/A           | N/A                  | N/A                       | N/A                       | N/A                    | index is out of bounds for axis 0 with size 0 |


*Generated on: 2025-04-12 18:56:25*