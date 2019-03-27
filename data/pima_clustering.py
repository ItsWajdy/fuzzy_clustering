from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data_into_array():
	features = []
	labels = []

	with open('pima/PimaIndians.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			tmp_list = [float(row['pregnant']), float(row['glucose']), float(row['diastolic']), float(row['triceps']),
						float(row['insulin']), float(row['bmi']), float(row['diabetes']), float(row['age'])]
			features.append(tmp_list)

			label = 0 if row['test'] == 'negatif' else 1
			labels.append([label])

	return np.array(features), np.array(labels)


def write_cmeans_to_csv(model):
	with open('pima/pima_cmeans_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('pima/pima_cmeans_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


def write_gk_to_csv(model):
	with open('pima/pima_gk_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('pima/pima_gk_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


def plot(model, data):
	Z = data.transpose()
	dim = data.shape[1]
	cluster1 = []
	cluster2 = []
	for k in range(Z.shape[1]):
		sample = np.reshape(Z[:, k], [Z.shape[0], 1])
		pred = np.argmax(model.predict(sample)) + 1
		if pred == 1:
			cluster1.append(sample)
		elif pred == 2:
			cluster2.append(sample)

	cluster1 = np.array(cluster1).transpose()
	cluster2 = np.array(cluster2).transpose()

	for i in range(dim):
		for j in range(dim):
			if i == j:
				continue

			plt.subplot(dim, dim, i*dim + j + 1)
			plt.scatter(cluster1[:, i], cluster1[:, j], c='r')
			plt.scatter(cluster2[:, i], cluster2[:, j], c='g')

	plt.show()


data, labels = read_data_into_array()
Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 2)
gk.fit(Z, 2)

plot(c_means, data)
plot(gk, data)

write_cmeans_to_csv(c_means)
write_gk_to_csv(gk)
