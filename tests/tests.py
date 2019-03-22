from fuzzy_clustering.c_means import Model
import numpy as np
import matplotlib.pyplot as plt
import random


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

model.fit(Z, 2, 2, 0.01)

plt.scatter(Z[0, :], Z[1, :])
plt.scatter(model.V[0, :], model.V[1, :], marker='x', c='r')
plt.show()

points = []
for i in range(50):
	x = random.uniform(0, 4)
	y = random.uniform(6, 10)
	points.append([x, y])

for i in range(50):
	x = random.uniform(6, 10)
	y = random.uniform(0, 4)
	points.append([x, y])

Z = np.array(points).transpose()
model = Model()
model.fit(Z, 2, 2, 0.01)

plt.scatter(Z[0, :], Z[1, :])
plt.scatter(model.V[0, :], model.V[1, :], marker='x', c='r')
plt.show()
