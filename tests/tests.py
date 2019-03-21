from fuzzy_clustering.c_means import Model
import numpy as np


model = Model()
Z = np.array([
	[1, 2],
	[2, 1],
	[2, 3],
	[3, 2],
	[5, 2],
	[5, 5],
	[7, 2],
	[8, 1],
	[8, 3],
	[9, 2]
]).transpose()

model.fit(Z, 2, 2, 1)
