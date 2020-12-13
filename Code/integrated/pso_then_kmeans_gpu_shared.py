import random
from matplotlib import pyplot as plt
from numpy import random as rand
import numpy as np
import math
from random import random, shuffle, gauss, sample, seed
import random as rd
from matplotlib import pyplot as plt
from numba import njit, cuda
import numpy
import numba
from numba import njit
from numpy import inf
import time 
import config

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

def myCluster(points, K, pointsAlreadyOnGPU = False, centroids = []):
    blockdim = 16
    griddim = 1 + (len(points) - 1)//blockdim

    #Your kmeans code will go here to cluster given points in K clsuters. If visuals = True, the code will also plot graphs to show the current state of clustering
    if not len(centroids):
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
    rd.seed(seed)
    np.random.seed(seed)
    if visuals:
        plt.scatter(points[:, 0], points[:, 1], color='red', alpha = 0.1, edgecolor='blue')
        plt.title("INITIAL POINTS")
        plt.show()
    clusters, centroids = runKMeansGPU(points, K, visuals = visuals, given_centroids =  given_centroids)
    print ("The score of best Kmeans clustering is:", clusterQuality(points, clusters, centroids))

@njit
def my_inf():
    return inf
    
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

# Shared memory fitness calculation
# In a single phase each block imports datapoints into shared memory equal to BlockDim and calculates
# distane from all the imported data points

@cuda.jit
def fitness_GPU(particles_pos, data, num_particles, particle_fitness):
    index = cuda.grid(1)
    tempData = cuda.shared.array(shape=(BLOCKDIM, DATADIM), dtype=numba.float64)
    data_dim = data.shape[1]

    if index < num_particles:
        particle = particles_pos[index]
        
    sum_dists = 0

    phases = (data.shape[0] - 1) // BLOCKDIM + 1
    for phase in range(phases):
        if phase * BLOCKDIM + cuda.threadIdx.x < data.shape[0]:
            for d in range(DATADIM):
                tempData[cuda.threadIdx.x, d] = data[phase * BLOCKDIM + cuda.threadIdx.x, d]
        else:
            for d in range(DATADIM):
                tempData[cuda.threadIdx.x, d] = 0

        cuda.syncthreads()

        if index < num_particles:
            for point in range(tempData.shape[0]):
                if point + BLOCKDIM * phase < data.shape[0]:
                    min_dist = my_inf()
                    for centroid in range(0, particle.shape[0], data_dim):
                        dist = 0
                        for k in range(data_dim):
                            dist += (tempData[point, k] - particle[centroid + k]) ** 2
                        if(dist < min_dist):
                            min_dist = dist
                    sum_dists += min_dist

        cuda.syncthreads()

    if index < num_particles:
        particle_fitness[index] = sum_dists #/ data.shape[0]
    

@cuda.jit
def update_GPU(particles_pos, particles_vel, particles_best_pos, global_best_pos, w, c1, c2, random_numbers, num_particles):
    start = cuda.grid(1)
    stride = cuda.blockDim.x * cuda.gridDim.x;
    for i in range(start, num_particles, stride):
        if(i < num_particles):
            r1 = random_numbers[i * 2]
            r2 = random_numbers[i * 2 + 1]
            for d in range(len(particles_pos[i])):
                particles_vel[i][d] = w * particles_vel[i][d] + c1 * r1 * (particles_best_pos[i][d] - particles_pos[i][d]) + c2 * r2 * (global_best_pos[d] - particles_pos[i][d])
                particles_pos[i][d] += particles_vel[i][d]

@cuda.jit
def find_max(fitness, x):
    x[0] = fitness.argmax()

def main_GPU(
        data,
        datadim,
        n = 10000,
        particles = 10000,
        iterations = 100,
        c = 3,
        w = 0.99,
        c1 = 0.3,
        c2 = 0.3,
        blockdim = 64
    ):
    initStart = time.time()
    griddim = (particles - 1) // blockdim + 1
     


    particles_pos, particles_vel = init_particles(particles, c, data)
    particles_pos = numpy.array(particles_pos)
    particles_vel = numpy.array(particles_vel)

    initEnd = time.time()
    initTime = round(initEnd - initStart, 3)
    
    particles_best_pos = [i.copy() for i in particles_pos]
    particles_best_fit = [float('inf') for i in range(len(particles_pos))]
    particles_best_pos = numpy.array(particles_best_pos)
    particles_best_fit = numpy.array(particles_best_fit)
    
    global_best_fit = float('inf')
    global_best_pos = None
    global_best_index = None
    
    totalFitnessAvg = 0
    totalUpdateAvg = 0

    data_gpu = cuda.to_device(data)
    particles_pos_gpu = cuda.to_device(particles_pos)
    particles_vel_gpu = cuda.to_device(particles_vel)
    particles_best_pos_gpu = cuda.to_device(particles_best_pos)
    particles_best_fit_gpu = cuda.to_device(particles_best_fit)

    fitness_cpu = numpy.arange(particles).astype("float")
    fitness_gpu = cuda.to_device(fitness_cpu)
    
    for iter in range(iterations):
        totalFitnessStart = time.time()
        fitness_GPU[griddim, blockdim](particles_pos_gpu, data_gpu, particles, fitness_gpu)
        numba.cuda.synchronize()
        fitness_gpu.copy_to_host(fitness_cpu)

        random_numbers = []
        for p in range(len(particles_pos)):
            fitness_p = fitness_cpu[p]
            if(fitness_p < particles_best_fit[p]):
                particles_best_fit[p] = fitness_p
                particles_best_pos[p] = particles_pos[p].copy()
            if(fitness_p < global_best_fit):
                global_best_fit = fitness_p
                global_best_pos = particles_pos[p].copy()
                global_best_index = p
            random_numbers.append(random())
            random_numbers.append(random())

        totalFitnessEnd = time.time() - totalFitnessStart
        totalFitnessAvg += round(totalFitnessEnd, 3)

        totalUpdateStart = time.time()
        

        random_numbers_gpu = cuda.to_device(numpy.array(random_numbers))
        global_best_pos_gpu = cuda.to_device(global_best_pos)
        particles_best_pos_gpu = cuda.to_device(particles_best_pos)
        
        update_GPU[griddim, blockdim](particles_pos_gpu, particles_vel_gpu, particles_best_pos_gpu, global_best_pos_gpu, w, c1, c2, random_numbers_gpu, particles)
        numba.cuda.synchronize()

        particles_pos_gpu.copy_to_host(particles_pos)
        totalUpdateEnd = time.time() - totalUpdateStart
        totalUpdateAvg += round(totalUpdateEnd, 3)
    
    totalFitnessAvg = round(totalFitnessAvg/iterations, 3)
    totalUpdateAvg = round(totalUpdateAvg/iterations, 3)
    fitnessPerParticle = round(totalFitnessAvg/particles, 3)
    updatePerParticle = round(totalUpdateAvg/particles, 3)



    for i in range(global_best_index, global_best_index + 1):
        l = []
        for point in data:
            min_dist = float('inf')
            min_value = None
            for centroid in range(0, len(particles_best_pos[i]), len(data[0])):
                dist = 0
                for k in range(len(data[0])):
                    dist += (point[k] - particles_best_pos[i][centroid + k]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_value = centroid // len(data[0])
            l.append(min_value)
    return global_best_pos, data_gpu

def main(
        data,
        c,
        random_state,
        DATADIM = 2,
        BLOCKDIM = 64,
        particles = 10,
        iterations = 100,
        w = 0.99,
        c1 = 0.15,
        c2 = 0.2,
    ):
    start = time.time()
    rd.seed(random_state)
    np.random.seed(random_state)
    centroids, points = main_GPU(data, DATADIM, blockdim = BLOCKDIM, particles=particles, iterations=iterations, c=c, w=w, c1=c1, c2=c2)
    centroids = centroids.reshape((c, DATADIM))
    
    #running kmeans
    main_kmeans_gpu(points, c, random_state, visuals = True, given_centroids = centroids)
    end = time.time()
    print(f"time taken: {end - start}") 
    return 