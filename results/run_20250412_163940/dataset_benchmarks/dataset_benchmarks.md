# Dataset Benchmarks

| dataset                |   original_vertices |   original_edges |   sampled_vertices |   sampled_edges |   density |   avg_degree |   lcc_size |   lcc_fraction | layout_time   |   degree_correlation | betweenness_correlation   |   eigenvector_correlation |   pagerank_correlation |
|:-----------------------|--------------------:|-----------------:|-------------------:|----------------:|----------:|-------------:|-----------:|---------------:|:--------------|---------------------:|:--------------------------|--------------------------:|-----------------------:|
| snap-facebook_combined |                4039 |            88234 |               1000 |            5345 |    0.0107 |       10.69  |        448 |          0.448 | 4.78s         |                0.546 | 0.538                     |                     0.088 |                  0.334 |
| snap-ca-GrQc           |               26197 |            14484 |               1000 |              22 |    0      |        0.044 |          3 |          0.003 | 7.42s         |                0.949 | N/A                       |                     0.107 |                  0.949 |
| snap-ca-HepTh          |               68746 |            25973 |               1000 |               7 |    0      |        0.014 |          2 |          0.002 | 5.19s         |               -0.007 | N/A                       |                    -0.019 |                 -0.007 |


*Generated on: 2025-04-12 16:40:58*