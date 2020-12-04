import random
from matplotlib import pyplot as plt
from numpy import random as rand
import numpy as np
import math
from numba import njit, cuda
import numpy
import numba

from random import random, shuffle, gauss, sample, seed
from sklearn import datasets
from sklearn.decomposition import PCA
import random 
import numpy as np 

def getPointsFromDataIris():
    # Iris
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    maxi = data.max(axis = 0)
    mini = data.min(axis = 0)
    data = (data - mini) / (maxi - mini)
    return data, labels, 3

def getPointsFromDataDigits():
    # Digits
    digits = datasets.load_digits()
    data = digits.data
    labels = digits.target

    n_digits = 10
    data = PCA(n_components=2).fit_transform(data)

    maxi = data.max(axis = 0)
    mini = data.min(axis = 0)
    diff = maxi - mini
    diff = (diff == 0) + diff
    data = (data - mini) / (diff)
    return data, labels, 10

def initializePoints(n, c = 3):
    l = [   
            [random.gauss(0.5, 0.1) + j, 
            random.gauss(0.5, 0.1)] 
            for j in range(c) for i in range(n)
        ]
    random.shuffle(l)
    return np.array(l)

def init_particles(n_particles, n_clusters, data):
    particles_pos = []
    particles_vel = []
    data = data.tolist()
    for i in range(n_particles):
        l2 = []
        clusters = sample(data, n_clusters)
        for cluster in clusters:
            l2.extend(cluster)
        particles_pos.append(l2)
        particles_vel.append([random() * 0.5 - 0.25 for i in range(n_clusters * len(data[0]))])
    return particles_pos, particles_vel

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

@cuda.jit
def calculateMeanNewClusters(points, cluster, centroids, pointsPerCluster):
    p = cuda.grid(1)
    if(p < points.shape[0]):
        for d in range(points.shape[1]):
            numba.cuda.atomic.add(centroids, (cluster[p], d), points[p][d])
        numba.cuda.atomic.add(pointsPerCluster, (cluster[p], 0), 1)
    
@cuda.jit
def findNearestCluster(points, centroidsGPU, clusterGPU):
    p = cuda.grid(1)
    if(p < points.shape[0]):
        minDistanceCentroid = -1
        minDistance = my_inf()
        for c in range(len(centroidsGPU)):
            distance = 0
            for d in range(points.shape[1]):
                distance += (points[p][d] - centroidsGPU[c][d]) ** 2
            if minDistance > distance:
                minDistance = distance
                minDistanceCentroid = c
        clusterGPU[p] = minDistanceCentroid

def myCluster(points, K, pointsAlreadyOnGPU = False, centroids = None):
    blockdim = 16
    griddim = 1 + (len(points) - 1)//blockdim

    #Your kmeans code will go here to cluster given points in K clsuters. If visuals = True, the code will also plot graphs to show the current state of clustering
    if not centroids:
        centroids = np.array([points[i] for i in rand.randint(0, len(points), K)]) #random centroids

    prevCentroids = np.ones(shape=centroids.shape) * float('inf')
    changeInCentroids = np.array([changeCPU(prevCentroids, centroids)])
    iteration = 0

    if pointsAlreadyOnGPU:
        pointsGPU = points
    else:
        pointsGPU = cuda.to_device(points)

    centroidsGPU = cuda.to_device(centroids)
    prevCentroidsGPU = cuda.to_device(prevCentroids)

    changeInCentroidsGPU =  cuda.device_array_like(changeInCentroids)
    
    clusters = np.zeros((len(points)), dtype=np.float64)
    clusterGPU = cuda.device_array_like(clusters)

    # pointsPerCluster = np.zeros((len(centroids), 1))
    # pointsPerClusterGPU = cuda.to_device(pointsPerCluster)

    while changeInCentroids[0] > 0.001: #stopping condition
        iteration += 1
        
        findNearestCluster[griddim, blockdim](pointsGPU, centroidsGPU, clusterGPU)
        numba.cuda.synchronize()
        
        prevCentroids = centroidsGPU.copy_to_host()
        
        # centroids = centroidsGPU.copy_to_host()

        # clusters = clusterGPU.copy_to_host()
        pointsPerCluster = np.zeros((len(centroids), 1), dtype=np.int32)
        pointsPerClusterGPU = cuda.to_device(pointsPerCluster)

        centroids = np.zeros_like(centroids, dtype=np.float32)
        centroidsGPU = cuda.to_device(centroids)

        calculateMeanNewClusters[griddim, blockdim](pointsGPU, clusterGPU, centroidsGPU, pointsPerClusterGPU)
        numba.cuda.synchronize()

        # centroidsGPU = cuda.to_device(centroids)
        centroids = centroidsGPU.copy_to_host()
        pointsPerCluster = pointsPerClusterGPU.copy_to_host()
        centroids = centroids / np.maximum(pointsPerCluster, 1)

        # centroids = centroids * (pointsPerCluster != 0)

        centroids = centroids + prevCentroids * (pointsPerCluster == 0)


        # clusters = clusterGPU.copy_to_host()
        # plt.scatter(points[:, 0], points[:, 1], c = clusters)
        # plt.plot(centroids[:, 0], centroids[:, 1], "kX")
        # plt.show()

        centroidsGPU = cuda.to_device(centroids)

        changeInCentroids[0] = changeCPU(prevCentroids, centroids)

    clusters = clusterGPU.copy_to_host()
    return clusters, centroids

def SSE(points, clusters, centroids):
    distances = np.array([points[i] - centroids[int(clusters[i])] for i in range(len(points))])
    squaredDistances = np.array([np.linalg.norm(distances[i])**2 for i in range(len(clusters))])
    return np.sum(squaredDistances)

def clusterQuality(points, clusters, centroids):
    score = SSE(points, clusters, centroids)
    return score

def runKMeansGPU(points, K, given_centroids = None, N = 1, visuals = False):
    clusters = None
    N = 1
    minimumScore, minimumScoreCluster = math.inf, None
    clusters, centroids = myCluster(points, K, centroids = given_centroids)
    if visuals:
        plt.scatter(points[:, 0], points[:, 1], c = clusters)
        plt.title("{0} points clustered into {1} clusters".format(len(points), K))
        plt.savefig(f"../../figures/KMEANS_GPU_{len(points)}_points_{K}_clusters.jpg")
        plt.show()
    return clusters, centroids

def main_kmeans_gpu(points, K, seed, given_centroids = None, visuals = False):
    np.random.seed(seed)
    random.seed(seed)
    if visuals:
        plt.scatter(points[:, 0], points[:, 1], color='red', alpha = 0.1, edgecolor='blue')
        plt.title("INITIAL POINTS")
        plt.show()
    clusters, centroids = runKMeansGPU(points, K, visuals = visuals, given_centroids =  given_centroids)
    print ("The score of best Kmeans clustering is:", clusterQuality(points, clusters, centroids))



seed = 20
np.random.seed(seed)
random.seed(seed)

K = 5
N = 2000
points = initializePoints(N, K)
main_kmeans_gpu(points, K, seed, visuals = False)