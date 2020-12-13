import random
from numpy import random as rand
import numpy as np
import math

from random import random, shuffle, gauss, sample, seed
import random 
import numpy as np 


def initializePoints(n, c = 3):
    l = [   
            [random.gauss(0.5, 0.1) + j, 
            random.gauss(0.5, 0.1)] 
            for j in range(c) for i in range(n)
        ]
    random.shuffle(l)
    return np.array(l)

def change(prev, current): #to compute change in previous and current centroids
    prev = np.array(prev); current = np.array(current)
    return np.linalg.norm(prev-current)

def calcDistance(point, centroid):
    temp = 0.
    for d in range(len(point)):
        temp += ( (point[d] - centroid[d]) ** 2 )
    return temp ** 0.5 

def calculateCluster(points, centroids):
    cluster = dict()
    
    for c in range(len(centroids)):
        cluster[c] = []

    for p in range(len(points)):
        min_cent = -1 
        min_dist = math.inf
        for c in range(len(centroids)):
            dist = calcDistance(points[p], centroids[c]) 
            if dist < min_dist:
                min_dist = dist
                min_cent = c
        cluster[min_cent].append(points[p])
    return cluster

def calculateNewCentroids(cluster):
    centroids = []
    for c in range(len(cluster)):
        mean_c = [0, 0]
        count = 0
        for p in range(len(cluster[c])):
            count += 1
            mean_c[0] += cluster[c][p][0]
            mean_c[1] += cluster[c][p][1]
        mean_c[0] /= count
        mean_c[1] /= count
        centroids.append(mean_c)
    return np.array(centroids)

def cluster(points,K,visuals = True):
    clusters=[]
    #Your kmeans code will go here to cluster given points in K clsuters. 
    #If visuals = True, the code will also plot graphs to show the current state of clustering
    centroids = np.array([points[i] for i in rand.randint(0, len(points), K)]) #random centroids
    prevCentroids = np.zeros(shape=centroids.shape)
    changeInCentroids = change(prevCentroids, centroids)
    iteration = 0
    while changeInCentroids > 0: #stopping condition
        iteration += 1
        changeInCentroids = change(prevCentroids, centroids)
        
        cluster = calculateCluster(points, centroids)
        
        prevCentroids = centroids.copy()
        centroids = np.zeros(centroids.shape)

        centroids = calculateNewCentroids(cluster)
        clusters = [cluster[i] for i in range(K)]

    return np.array(clusters), np.array(centroids)


def SSE(points, clusters, centroids):
    distances = np.array([clusters[i] - centroids[i] for i in range(len(clusters))])
    squaredDistances = np.array([np.linalg.norm(distances[i])**2 for i in range(len(clusters))])
    return np.sum(squaredDistances)

def clusterQuality(points, clusters, centroids):
    score = SSE(points, clusters, centroids)
    return score

def runKMeansCPU(points, K, given_centroids = None, N = 1, visuals = False):
    clusters = []
    minimumScore, minimumScoreCluster = math.inf, None
    for i in range(N):
        clusters, centroids = cluster(points, K, given_centroids)
        
    colors = []
    for i in points:
        colors.append(((centroids - i) ** 2).sum(axis = 1).argmin())
    if visuals:
        plt.scatter(points[:, 0], points[:, 1], c= colors)
        plt.title("{0} points clustered into {1} clusters".format(len(points), K))
        plt.savefig(f"../../figures/KMEANS_CPU_{len(points)}_points_{K}_clusters.jpg")
        plt.show()
    return clusters, centroids

def main_kmeans_cpu(points, K, seed, centroids = None, visuals = False):
    np.random.seed(seed)
    random.seed(seed)
    if visuals:
        plt.scatter(points[:, 0], points[:, 1], color='red', alpha = 0.1, edgecolor='blue')
        plt.title("INITIAL POINTS")
        plt.show()
    clusters, centroids = runKMeansCPU(points, K, given_centroids = centroids, visuals = visuals)
    print ("The score of best Kmeans clustering is:", clusterQuality(points, clusters, centroids))

if __name__ == "__main__":
    seed = 20
    np.random.seed(seed)
    random.seed(seed)

    K = 5
    N = 2000
    points = initializePoints(N, K)
    main_kmeans_cpu(points, K, seed, visuals = False)

