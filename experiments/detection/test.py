import json
import numpy as np


cube = np.load("TC_time_space_cube.npy")
freq = cube.sum(axis=0)

print("All non-zero grid cells:")
print(np.argwhere(freq > 0))

print("Max freq:", freq.max())
