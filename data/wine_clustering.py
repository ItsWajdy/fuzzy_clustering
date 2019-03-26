from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK
import csv
import numpy as np


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


# def compare(labels, predictions):
# 	acc = 0
# 	for i in range(labels.shape[0]):
# 		if labels[i] == predictions[i]:
# 			acc += 1
# 	acc /= labels.shape[0]
# 	return acc
#
#
# def evaluate(data, labels, model):
# 	predictions = model.predict(data)
# 	pred = []
#
# 	for k in range(predictions.shape[1]):
# 		pred.append(np.argmax(predictions[:, k]) + 1)
# 	return compare(labels, np.array(pred))


# write_data_to_csv()
data, labels = read_data_into_array()
Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 3)
gk.fit(Z, 3)
