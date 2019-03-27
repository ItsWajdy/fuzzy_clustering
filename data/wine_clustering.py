from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_original_file():
	with open('wine/wine.data', 'rb') as file:
		lines = file.readlines()
		formatted = []
		for line in lines:
			add = ''
			for char in line:
				if chr(char) != '\n':
					add += chr(char)
			formatted.append(add)

	return formatted


def write_data_to_csv():
	lines = read_original_file()
	data_str = []
	for line in lines:
		data_points = line.split(',')
		data_str.append(data_points)

	with open('wine/wine.csv', 'w', newline='') as file:
		field_names = ['alcohol', 'malic acid', 'ash', 'ash alcalinity', 'magnesium', 'total phenols', 'falvanoids',
					   'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'od280/od315', 'proline', 'class']

		writer = csv.DictWriter(file, fieldnames=field_names)
		writer.writeheader()

		for sample in data_str:
			writer.writerow({
				'alcohol': float(sample[1]),
				'malic acid': float(sample[2]),
				'ash': float(sample[3]),
				'ash alcalinity': float(sample[4]),
				'magnesium': float(sample[5]),
				'total phenols': float(sample[6]),
				'falvanoids': float(sample[7]),
				'nonflavanoid phenols': float(sample[8]),
				'proanthocyanins': float(sample[9]),
				'color intensity': float(sample[10]),
				'hue': float(sample[11]),
				'od280/od315': float(sample[12]),
				'proline': float(sample[13]),
				'class': float(sample[0])
			})


def read_data_into_array():
	features = []
	labels = []

	with open('wine/wine.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			tmp_list = [float(row['alcohol']), float(row['malic acid']), float(row['ash']), float(row['ash alcalinity']),
						float(row['magnesium']), float(row['total phenols']), float(row['falvanoids']),
						float(row['nonflavanoid phenols']), float(row['proanthocyanins']), float(row['color intensity']),
						float(row['hue']), float(row['od280/od315']), float(row['proline'])]
			features.append(tmp_list)

			label = int(float(row['class']))
			labels.append([label])

	return np.array(features), np.array(labels)


def write_cmeans_to_csv(model):
	with open('wine/wine_cmeans_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('wine/wine_cmeans_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


def write_gk_to_csv(model):
	with open('wine/wine_gk_U.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(model.U.shape[0]):
			writer.writerow(model.U[i, :])

	with open('wine/wine_gk_V.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=' ',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)

		for i in range(model.V.shape[0]):
			writer.writerow(model.V[i, :])


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


# write_data_to_csv()
data, labels = read_data_into_array()
Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 3)
gk.fit(Z, 3)

plot(c_means, data)
plot(gk, data)

write_cmeans_to_csv(c_means)
write_gk_to_csv(gk)
