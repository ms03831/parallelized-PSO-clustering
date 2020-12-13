import math
from random import random, shuffle, gauss, sample, seed
from matplotlib import pyplot as plt
from numba import cuda
import numpy
import numba
from numba import njit
from numpy import inf
import time 

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

@cuda.jit
def fitness_GPU(particles_pos, data, num_particles, particle_fitness):
    index = cuda.grid(1)
    if index < num_particles:
        data_dim = data.shape[1]
        particle = particles_pos[index]
        sum_dists = 0
        for point in range(data.shape[0]):
            min_dist = my_inf()
            for centroid in range(0, particle.shape[0], data_dim):
                dist = 0
                for k in range(data_dim):
                    dist += (data[point][k] - particle[centroid + k]) ** 2
                if(dist < min_dist):
                    min_dist = dist
            sum_dists += min_dist
        particle_fitness[index] = sum_dists

@cuda.jit
def update_GPU(particles_pos, particles_vel, particles_best_pos, global_best_pos, w, c1, c2, random_numbers, num_particles):
    i = cuda.grid(1)
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
    return l
    
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
    centroids = main_GPU(data, DATADIM, blockdim = BLOCKDIM, particles=particles, iterations=iterations, c=c, w=w, c1=c1, c2=c2)
    return centroids