import os
import numpy as np


# to clean txt files of datasets: should be single space delimited
for file in os.listdir("../datasets/"):
    print("open "+file)
    filepath = os.path.join("../datasets", file)
    mat = np.loadtxt(filepath)
    np.savetxt(filepath, mat, fmt='%d')
    print(file+" finished")