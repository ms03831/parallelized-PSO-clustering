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