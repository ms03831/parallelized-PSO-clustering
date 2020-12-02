import random
from matplotlib import pyplot as plt
from numpy import random as rand
import numpy as np
import math
from numba import njit, cuda
import numpy
import numba

@cuda.jit(device=True)
def my_inf():
    return np.inf

def initializePoints(n, c = 3):
    l = [   
            [random.gauss(0.5, 0.1) + j, 
            random.gauss(0.5, 0.1)] 
            for j in range(c) for i in range(n)
        ]
    random.shuffle(l)
    return np.array(l)

def changeCPU(prev, current): #to compute change in previous and current centroids
    prev = np.array(prev); current = np.array(current)
    return np.linalg.norm(prev-current)

def change(prev, current, changeInCentroidsGPU): #to compute change in previous and current centroids
    temp = 0.
    for i in range(prev.shape[0]):
        temp += ( (prev[i][0] - current[i][0]) ** 2 + 
                  (prev[i][1] - current[i][1]) ** 2   )
    changeInCentroidsGPU[0] = temp ** 0.5

def calculateMeanNewClusters(centroidsGPU, clusterGPU):
    for i in range(centroidsGPU.shape[0]):
        pointcount = 0
        temp = [0, 0]
        for j in range(clusterGPU.shape[0]):
            if clusterGPU[j][0] == i:
                pointcount += 1
                temp[0] += clusterGPU[j][1]
                temp[1] += clusterGPU[j][2]
        if pointcount:
            centroidsGPU[i][0] = temp[0]/pointcount
            centroidsGPU[i][1] = temp[1]/pointcount
    return centroidsGPU
    
@cuda.jit
def findNearestCluster(xPointsGPU, yPointsGPU, centroidsGPU, clusterGPU):
    p = cuda.grid(1)
    minDistanceCentroid = -1
    minDistance = my_inf()
    for c in range(len(centroidsGPU)):
        distance = ((xPointsGPU[p] - centroidsGPU[c][0]) ** 2 + 
                    (yPointsGPU[p] - centroidsGPU[c][1]) ** 2) ** 0.5
        if minDistance > distance:
            minDistance = distance
            minDistanceCentroid = c
    clusterGPU[p][0] = minDistanceCentroid

def myCluster(points, K, visuals = True):
    blockdim = 16
    griddim = 1 + (len(points) - 1)//blockdim

    #Your kmeans code will go here to cluster given points in K clsuters. If visuals = True, the code will also plot graphs to show the current state of clustering
    xPoints, yPoints = np.array([i[0] for i in points]), np.array([i[1] for i in points])
    centroids = np.array([points[i] for i in rand.randint(0, len(points), K)]) #random centroids
    prevCentroids = np.zeros(shape=(3,2))
    changeInCentroids = np.array([changeCPU(prevCentroids, centroids)])
    iteration = 0

    xPointsGPU = cuda.to_device(xPoints)
    yPointsGPU = cuda.to_device(xPoints)
    centroidsGPU = cuda.to_device(centroids)
    prevCentroidsGPU = cuda.to_device(prevCentroids)

    changeInCentroidsGPU =  cuda.device_array_like(changeInCentroids)
    
    clusters = np.zeros((len(points), 3))
    clusterGPU = cuda.device_array_like(clusters)
    print("HELLO ", changeInCentroids.shape)
    while changeInCentroids[0] > 0: #stopping condition
        iteration += 1
        changeInCentroids[0] = changeCPU(prevCentroidsGPU, centroids)
        
        findNearestCluster[griddim, blockdim](xPointsGPU, yPointsGPU, centroidsGPU, clusterGPU)
        numba.cuda.synchronize()
        
        prevCentroids = centroidsGPU.copy_to_host()
        prevCentroidsGPU = cuda.to_device(prevCentroids)
        
        centroids = centroidsGPU.copy_to_host()

        centroids = calculateMeanNewClusters(centroids, clusterGPU)        
        centroidsGPU = cuda.to_device(centroidsGPU)
    clusters = clusterGPU.copy_to_host()
    return clusters

def SSE(clusters):
    centroids = np.array([np.mean(clusters[i], axis = 0) for i in range(len(clusters))])
    distances = np.array([clusters[i] - centroids[i] for i in range(len(clusters))])
    squaredDistances = np.array([np.linalg.norm(distances[i])**2 for i in range(len(clusters))])
    return np.sum(squaredDistances)

def clusterQuality(clusters):
    score = SSE(clusters)
    return score

def runKMeans(points, K, N, visuals):
    clusters = None
    N = 1
    minimumScore, minimumScoreCluster = math.inf, None
    myCluster(points, K, visuals = False)
    #     centroids = np.array([np.mean(clusters[i], axis = 0) for i in range(len(clusters))])
    #     if ((i + 1) % 5 == 0):    
    #         plt.figure()
    #         for c in clusters:
    #             plt.scatter(*zip(*c), alpha = 0.4)
    #         plt.plot([centroids[i][0] for i in range(K)], [centroids[i][1] for i in range(K)], 'kX', markersize=10, label="clusters")
    #         plt.legend()
    #         plt.title("{0} points clustered into {1} clusters in iteration number {2}".format(len(points), K, i + 1))
    #         plt.show()
    #     score = clusterQuality(clusters)
    #     if score < minimumScore: 
    #         minimumScore = score
    #         mininumScoreCluster = clusters
    # return clusters

