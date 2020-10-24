
# Parallelized Hybrid PSO-KMeans Clustering Algorithm 
### Final Project Proposal - CS 432 GPU Accelerated Computing
##### Members:
1. Rayyan-ul-Haq ru03588
2. Mudasir Hanif Shaikh ms03831
3. Syed Ammar Ahmed sa04050

### Problem Definition and Applications
![Clustering](merge3cluster.jpg)
Clustering is a very important task in unsupervised Machine Learning. The aim of clustering is to group similar objects together based on a distance metric. It is used in a variety of problems and domains such as image processing, finance, market research etc. For example, clustering can be used by businesses to group their customers based on their likes/dislikes, purchase power, frequency of purchase etc. [1].

One of the most popular algorithms used for clustering is the K-means algorithm and therefore has been a focus of many researchers. But as with all algorithms, K-means is not perfect. Many fronts have been explored in recent years to improve both the accuracy and execution time of this algorithm. Several of these attempts involved using a well known Swarm Intelligence approach known as Particle Swarm Optimization (PSO). This project aims to utilize and parallelize a hybrid approach using both PSO and K-Means for data clustering. 


### Literature Review
PSO has been used in literature to solve the problem of initial cluster centers affecting the performance of K-Means algorithm. Zhao et. al proposed that K-Means algorithm could be improved by using PSO to generate the initial cluster centers and experimentally found out that the improved k-mean clustering algorithm has obvious advantages on execution time [2]. Merwe et. al also used the same idea and found out that PSO can be used to find the centroids of a user specified number of clusters and then extended to use K-means clustering to seed the initial swarm [3]. PSO can also be used to reform the clusters formed by K-Means. Both of these approaches have shown good potential [3]. 

Ahmadyfard et. al worked on the limitations on PSO and noted that PSO algorithm can successfully converge during the initial stages of a global search, but around global optimum, the search process becomes very slow [4]. K-Means however achieves faster convergence when it is near optimal solution. The accuracy of K-Means is also higher which is what makes it so popular. They proposed a hybrid algorithm called "PSO-KM" and experimentally argued that the hybrid approach has more potential than both K-Means and PSO [4]. 

### Performance Analysis
We will analyze the performance and results of our algorithm on not one but several clustering benchmark datasets found [here](http://cs.joensuu.fi/sipu/datasets/) and present our analysis on the improvement (if any) of the parallelized GPU version of the hybrid approach over the CPU version. 

### Research and Novelty
From our research and exploration so far, we have realized that the problem of using a hybrid approach in itself has room for more research, however, our aim is not to improve the algorithm but rather present a parallelized version for it, which as we understand has not yet been properly explored. Therefore, we believe this project has a lot of research potential and can lead to a publication. As the task is not very simple, and would require a lot of research we believe this also **justifies three members** in our team.

### Type of project
We believe that our projects falls in the category of "Implement a GPU version of some existing computationally-intensive CPU code." The justification for this is outlined in the previous sections on novelty, problem definition and analysis.

### Major goals of the project
1. Implement the CPU version of the hybrid approach
2. Parallelize the hybrid approach and implement the GPU version on pyCUDA or Numba.
3. Performance analysis (outlined above)

## References
[1] J. Oyelade et al., "Data Clustering: Algorithms and Its Applications," 2019 19th International Conference on Computational Science and Its Applications (ICCSA), Saint Petersburg, Russia, 2019, pp. 71-81, doi: 10.1109/ICCSA.2019.000-1.

[2] M. Zhao, H. Tang, J. Guo, and Y. Sun, “Data Clustering Using Particle Swarm Optimization,” _Lecture Notes in Electrical Engineering Future Information Technology_, pp. 607–612, 2014.

[3] D. V. D. Merwe and A. Engelbrecht, “Data clustering using particle swarm optimization,” _The 2003 Congress on Evolutionary Computation, 2003. CEC '03._ https://ieeexplore.ieee.org/document/1299577

[4] A. Ahmadyfard and H. Modares, "Combining PSO and k-means to enhance data clustering," _2008 International Symposium on Telecommunications_, Tehran, 2008, pp. 688-691, doi: 10.1109/ISTEL.2008.4651388.
