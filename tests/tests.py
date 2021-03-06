from fuzzy_clustering.c_means import Model
import numpy as np
import matplotlib.pyplot as plt
import random


# model = Model()
# Z = np.array([
# 	[1, 2],
# 	[2, 1],
# 	[2, 3],
# 	[3, 2],
# 	[5, 2],
# 	[5, 5],
# 	[7, 2],
# 	[8, 1],
# 	[8, 3],
# 	[9, 2]
# ]).transpose()
#
# model.fit(Z, 2, 2, 0.01, 'diagonal')
#
# plt.scatter(Z[0, :], Z[1, :])
# plt.scatter(model.V[0, :], model.V[1, :], marker='x', c='r')
# plt.show()

points = []
for i in range(100):
	x = random.uniform(0, 30)
	y = random.uniform(2, 4)
	points.append([x, y])

for i in range(100):
	x = random.uniform(30, 60)
	y = random.uniform(0, 2)
	points.append([x, y])

Z = np.array(points).transpose()
model = Model()
model.fit(Z, 2, 2, 0.01)

plt.scatter(Z[0, :], Z[1, :])
plt.scatter(model.V[0, :], model.V[1, :], marker='x', c='r')

x = np.arange(0, 60, 0.1)
y = np.arange(0, 5, 0.1)

U1 = np.zeros([x.shape[0], y.shape[0]])
for i in range(x.shape[0]):
	for j in range(y.shape[0]):
		U1[i][j] = model.predict(np.reshape(np.array([x[i], y[j]]), [2, 1]))[0]

plt.contour(x, y, U1.transpose())
plt.show()
