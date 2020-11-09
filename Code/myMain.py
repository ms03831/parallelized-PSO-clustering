import math
from random import random, shuffle, gauss, sample
from matplotlib import pyplot as plt


def random_data(n, c):
    l = [[gauss(0.5, 0.1) + j, gauss(0.5, 0.1)] for j in range(c) for i in range(n)]
    shuffle(l)
    return l


def init_particles(n_particles, n_clusters, data):
    particles_pos = []
    particles_vel = []
    for i in range(n_particles):
        l2 = []
        clusters = sample(data, n_clusters)
        for cluster in clusters:
            l2.extend(cluster)
        particles_pos.append(l2)
        particles_vel.append([random() * 0.5 - 0.25 for i in range(n_clusters * len(data[0]))])
    return particles_pos, particles_vel


def fitness(particles_pos, data, index):
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
        sum_dists += dist
    return sum_dists / len(data)


def update(particles_pos, particles_vel, particles_best_pos, global_best_pos, i, w, c1, c2):
    for d in range(len(particles_pos[i])):
        particles_vel[i][d] = w * particles_vel[i][d] + c1 * random() * (particles_best_pos[i][d] - particles_pos[i][d]) + c2 * random() * (global_best_pos[d] - particles_pos[i][d])
        particles_pos[i][d] += particles_vel[i][d]
    

def main():
    c = 3
    n = 100
    particles = 10
    iterations = 100
    w = 0.95
    c1 = 0.1
    c2 = 0.2

    data = random_data(n, c)
    particles_pos, particles_vel = init_particles(particles, c, data)
    particles_best_pos = [i.copy() for i in particles_pos]
    particles_best_fit = [float('inf') for i in range(len(particles_pos))]

    global_best_fit = float('inf')
    global_best_pos = None

    for iter in range(iterations):
        for p in range(len(particles_pos)):
            fitness_p = fitness(particles_pos, data, p)
            if(fitness_p < particles_best_fit[p]):
                particles_best_fit[p] = fitness_p
                particles_best_pos[p] = particles_pos[p].copy()
            if(fitness_p < global_best_fit):
                global_best_fit = fitness_p
                global_best_pos = particles_pos[p].copy()
    
        for p in range(len(particles_pos)):
            update(particles_pos, particles_vel, particles_best_pos, global_best_pos, p, w, c1, c2)
        
    for i in range(len(particles_pos)):
        l = []
        for point in data:
            min_dist = float('inf')
            min_value = None
            for centroid in range(0, len(particles_pos[i]), len(data[0])):
                dist = 0
                for k in range(len(data[0])):
                    dist += (point[k] - particles_pos[i][centroid + k]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_value = centroid // len(data[0])
            l.append(min_value)
        
        plt.scatter(*zip(*data), c = l)
        plt.show()    

main()

