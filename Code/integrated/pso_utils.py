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