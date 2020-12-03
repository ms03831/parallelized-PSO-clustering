from utils import getPointsFromDataIris, getPointsFromDataIris
from kmeans_cpu import runKMeans, initializePoints, clusterQuality
import matplotlib.pyplot as plt
import numpy as np
import random

ITER = 1

seed = 20
np.random.seed(seed)
random.seed(seed)

# K = 3
# N = 50
# points = initializePoints(N, K)

points, labels, K = getPointsFromDataDigits()
points = points[:, :2]

# plt.scatter(*zip(*points), color='red', alpha = 0.2, edgecolor='blue')
# plt.title("INITIAL POINTS")
# plt.show()

clustering = runKMeans(points,K,ITER,True)
print ("The score of best Kmeans clustering is:", clusterQuality(clustering))

plt.scatter(points[:, 0], points[:, 1], c = labels)
plt.show()
