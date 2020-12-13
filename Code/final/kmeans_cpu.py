import random
from matplotlib import pyplot as plt
from numpy import random as rand
import numpy as np
import math

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
        
        cluster = dict([(c, []) for c in range(len(centroids))])
        for p in range(len(points)):
            belongsTo = dict()
            minDistanceCentroid = np.argmin([calcDistance(points[p], c) 
                                             for c in centroids]) #index of nearest centroid
            #belongsTo[points[p]] = centroids[minDistanceCentroid] 
            cluster[minDistanceCentroid].append(points[p])
        prevCentroids = centroids.copy()
        centroids = np.zeros(centroids.shape)
        for i in range(K):
            centroids[i] = np.mean(cluster[i], axis = 0)
        clusters = [cluster[i] for i in range(K)]
        # for c in clusters:
        #     plt.scatter(*zip(*c), alpha = 0.4)
        # plt.plot([centroids[i][0] for i in range(K)], [centroids[i][1] for i in range(K)], 'kX', markersize=10, label="clusters")
        # plt.legend()
        # plt.title("{0} points clustered into {1} clusters in inner iteration number {2}".format(len(points), K, iteration))
        # plt.show()
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
        '''
        if ((i + 1) % 5 == 0):    
            plt.figure()
            for c in clusters:
                plt.scatter(*zip(*c), alpha = 0.4)
            plt.plot([centroids[i][0] for i in range(K)], [centroids[i][1] for i in range(K)], 'kX', markersize=10, label="clusters")
            plt.legend()
            plt.title("{0} points clustered into {1} clusters in iteration number {2}".format(len(points), K, i + 1))
            plt.show()
        score = clusterQuality(points, clusters, centroids)
        if score < minimumScore: 
            minimumScore = score
            mininumScoreCluster = clusters
        '''
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