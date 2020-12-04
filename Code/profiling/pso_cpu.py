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

import math
from random import random, shuffle, gauss, sample, seed
from matplotlib import pyplot as plt
from numba import cuda
import numpy
import numba
from numba import njit
from numpy import inf
import time 

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

def fitness_CPU(particles_pos, data, index):
    data_dim = len(data[0])
    particle = particles_pos[index]
    sum_dists = 0
    for point in range(len(data)):
        min_dist = float('inf')
        for centroid in range(0, len(particle), data_dim):
            dist = 0
            for k in range(data_dim):
                dist += (data[point][k] - particle[centroid + k]) ** 2
            if(dist < min_dist):
                min_dist = dist
        sum_dists += min_dist
    return sum_dists

def update_CPU(particles_pos, particles_vel, particles_best_pos, global_best_pos, i, w, c1, c2):
    r1, r2 = random(), random()
    for d in range(len(particles_pos[i])):
        particles_vel[i][d] = w * particles_vel[i][d] + c1 * r1 * (particles_best_pos[i][d] - particles_pos[i][d]) + c2 * r2 * (global_best_pos[d] - particles_pos[i][d])
        particles_pos[i][d] += particles_vel[i][d]

def main_CPU(
        data,
        n = 10000,
        particles = 10000,
        iterations = 100,
        c = 3,
        w = 0.99,
        c1 = 0.3,
        c2 = 0.3
    ):
    initStart = time.time()

    particles_pos, particles_vel = init_particles(particles, c, data)
    particles_best_pos = [i.copy() for i in particles_pos]
    particles_best_fit = [float('inf') for i in range(len(particles_pos))]
    particles_pos = numpy.array(particles_pos)

    initEnd = time.time()
    initTime = round(initEnd - initStart, 3)

    global_best_fit = float('inf')
    global_best_pos = None
    global_best_index = -1

    totalFitnessAvg = 0
    totalUpdateAvg = 0

    for iter in range(iterations):
        totalFitnessStart = time.time()
        for p in range(len(particles_pos)):
            fitness_p = fitness_CPU(particles_pos, data, p)
            if(fitness_p < particles_best_fit[p]):
                particles_best_fit[p] = fitness_p
                particles_best_pos[p] = particles_pos[p].copy()
            if(fitness_p < global_best_fit):
                global_best_fit = fitness_p
                global_best_pos = particles_pos[p].copy()
                global_best_index = p
        totalFitnessEnd = time.time() - totalFitnessStart
        totalFitnessAvg += round(totalFitnessEnd, 3)

        totalUpdateStart = time.time()
        
        for p in range(len(particles_pos)):
            update_CPU(particles_pos, particles_vel, particles_best_pos, global_best_pos, p, w, c1, c2)
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
        
        plt.scatter(*zip(*data), c = l)
        plt.show()    
    return (initTime, totalFitnessAvg, totalUpdateAvg, fitnessPerParticle, updatePerParticle)

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
    seed(random_state)
    initTime, totalFitnessAvg, totalUpdateAvg, fitnessPerParticle, updatePerParticle = main_CPU(data, particles=particles, iterations=iterations, c=c, w=w, c1=c1, c2=c2)
    return (initTime, totalFitnessAvg, totalUpdateAvg, fitnessPerParticle, updatePerParticle)

#seeding and data generation
random_state = 20
np.random.seed(random_state)
random.seed(random_state)

K = 3
N = 500
points = initializePoints(N, K)

particles = 50
iterations = 100
w = 0.99
c1 = 0.15
c2 = 0.2
BLOCKDIM = 64
DATADIM = 2
main(   points, K, random_state, DATADIM, 
        BLOCKDIM, particles, iterations, 
        w, c1, c2   )