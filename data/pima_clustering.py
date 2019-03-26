from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK
import csv
import numpy as np


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


def evaluate(labels, predictions):
	acc = 0
	for i in range(labels.shape[0]):
		if labels[i] == predictions[i]:
			acc += 1
	acc /= labels.shape[0]
	return acc


data, labels = read_data_into_array()
Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 2, 3, 0.00001)
gk.fit(Z, 2, 3, 0.00001)
