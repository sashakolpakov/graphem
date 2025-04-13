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

| dataset                |   original_vertices |   original_edges |   sampled_vertices |   sampled_edges |   density |   avg_degree |   lcc_size |   lcc_fraction | layout_time   |   degree_correlation | betweenness_correlation   |   eigenvector_correlation |   pagerank_correlation |
|:-----------------------|--------------------:|-----------------:|-------------------:|----------------:|----------:|-------------:|-----------:|---------------:|:--------------|---------------------:|:--------------------------|--------------------------:|-----------------------:|
| snap-facebook_combined |                4039 |            88234 |               1000 |            5345 |    0.0107 |       10.69  |        448 |          0.448 | 5.19s         |                0.546 | 0.538                     |                     0.088 |                  0.334 |
| snap-ca-GrQc           |               26197 |            14484 |               1000 |              22 |    0      |        0.044 |          3 |          0.003 | 6.24s         |                0.949 | N/A                       |                     0.107 |                  0.949 |
| snap-ca-HepTh          |               68746 |            25973 |               1000 |               7 |    0      |        0.014 |          2 |          0.002 | 4.07s         |               -0.007 | N/A                       |                    -0.019 |                 -0.007 |


*Generated on: 2025-04-12 17:03:55*