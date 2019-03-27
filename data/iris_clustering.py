from sklearn.datasets import load_iris
from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK
import matplotlib.pyplot as plt
import numpy as np
import csv


def plot(model, data):
	Z = data.transpose()
	dim = data.shape[1]
	cluster1 = []
	cluster2 = []
	cluster3 = []
	for k in range(Z.shape[1]):
		sample = np.reshape(Z[:, k], [Z.shape[0], 1])
		pred = np.argmax(model.predict(sample)) + 1
		if pred == 1:
			cluster1.append(sample)
		elif pred == 2:
			cluster2.append(sample)
		else:
			cluster3.append(sample)

	cluster1 = np.array(cluster1).transpose()
	cluster2 = np.array(cluster2).transpose()
	cluster3 = np.array(cluster3).transpose()

	for i in range(dim):
		for j in range(dim):
			if i == j:
				continue

			plt.subplot(dim, dim, i*dim + j + 1)
			plt.scatter(cluster1[:, i], cluster1[:, j], c='r')
			plt.scatter(cluster2[:, i], cluster2[:, j], c='g')
			plt.scatter(cluster3[:, i], cluster3[:, j], c='b')

	plt.show()


def plot_contour(model, data):
	Z = data.transpose()
	dim = data.shape[1]
	cluster1 = []
	cluster2 = []
	cluster3 = []
	for k in range(Z.shape[1]):
		sample = np.reshape(Z[:, k], [Z.shape[0], 1])
		pred = np.argmax(model.predict(sample)) + 1
		if pred == 1:
			cluster1.append(sample)
		elif pred == 2:
			cluster2.append(sample)
		else:
			cluster3.append(sample)

	cluster1 = np.array(cluster1).transpose()
	cluster2 = np.array(cluster2).transpose()
	cluster3 = np.array(cluster3).transpose()

	plt.scatter(cluster1[:, 0], cluster1[:, 1], c='r')
	plt.scatter(cluster2[:, 0], cluster2[:, 1], c='g')
	plt.scatter(cluster3[:, 0], cluster3[:, 1], c='b')

	x = np.arange(int(np.min(data[:, 0])), int(np.max(data[:, 0])), 0.025)
	y = np.arange(int(np.min(data[:, 1])), int(np.max(data[:, 1])), 0.025)
	z = np.arange(int(np.min(data[:, 2])), int(np.max(data[:, 2])), 0.025)
	w = np.arange(int(np.min(data[:, 3])), int(np.max(data[:, 3])), 0.025)

	U1 = np.zeros([x.shape[0], y.shape[0]])
	for i in range(x.shape[0]):
		for j in range(y.shape[0]):
			U1[i][j] = model.predict(np.reshape(np.array([x[i], y[j], 0, 0]), [4, 1]))[0]

	plt.contour(x, y, U1.transpose())
	plt.show()


def write_cmeans_to_csv(model):
	with open('iris/iris_cmeans_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('iris/iris_cmeans_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


def write_gk_to_csv(model):
	with open('iris/iris_gk_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('iris/iris_gk_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


iris = load_iris()
data = iris.data
labels = iris.target

Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 3)
gk.fit(Z, 3)

plot(c_means, data)
plot(gk, data)

write_cmeans_to_csv(c_means)
write_gk_to_csv(gk)

plot_contour(gk, data)
