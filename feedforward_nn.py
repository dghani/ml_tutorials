import numpy as np 
from pprint import pprint

class NeuralNet(object): 
    RNG = np.random.default_rng() 

    def __init__(self, topology:list[int] = []): 
        self.topology = topology 
        self.weight_mats = []         
        self._init_matrices() 

    def _init_matrices(self):
        if len(self.topology) > 1: 
            j = 1 
            for i in range(len(self.topology)-1): 
                num_rows = self.topology[i] 
                num_cols = self.topology[j]                 
                mat = self.RNG.random(size=(num_rows, num_cols))
                self.weight_mats.append(mat) 
                j += 1
    
    def evaluate(self, x: np.ndarray):
        curarray = x
        for i in range(len(self.topology)-1): 
            curarray = np.dot(curarray, self.weight_mats[i])
            curarray = self.relu(curarray) # transfer function
            pprint(f"{curarray=}")

    # replace all negative array values with 0. 
    def relu(self, x: np.ndarray):
        for i, val in enumerate(x):
            if val < 0:
                x[i] = 0
        return x
        

a = NeuralNet([3, 4, 2])

pprint(f"{a.topology=}")
pprint(f"{a.weight_mats=}")

print("evaluate:")
a.evaluate([-1, 2, -3])