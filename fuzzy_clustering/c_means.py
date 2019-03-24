import numpy as np
import random


class Model:
	def __init__(self):
		self.c = 2
		self.N = 0
		self.n = 0
		self.m = None
		self.epsilon = None
		self.A = None
		self.V = None
		self.U = None

	def fit(self,
			Z,
			c,
			fuzziness_parameter=2,
			termination_criterion=0.01,
			norm_inducing_matrix='identity'):
		"""
		:param Z: Training instances to cluster
		:param c: Number of cluster
		:param fuzziness_parameter: The weighting exponent m influencing the fuzziness of the clusters
			As m approaches 1, the partition becomes hard. As m approaches inf, the partition becomes completely fuzzy
		:param termination_criterion: The c-means algorithm terminates when the difference between U in two successive
			iterations is smaller the this (epsilon)
		:param norm_inducing_matrix: The shape of the clusters is determined by the choice of the norm-inducing matrix A
			Only 3 options for this parameter are permitted, 'identity', 'diagonal', and 'mahalonobis'
		:return: None
		"""

		self.__init_vars(Z, c, fuzziness_parameter, termination_criterion, norm_inducing_matrix)
		prev_U = 0
		first_time_through = True
		while first_time_through or not self.__reached_termination(prev_U):
			first_time_through = False
			self.__compute_cluster_means(Z)
			D = self.__compute_distances(Z)

			prev_U = np.zeros([self.c, self.N])
			for i in range(self.c):
				for k in range(self.N):
					prev_U[i][k] = self.U[i][k]

			self.__update_partition_matrix(D, Z)
		# TODO: remove asserts in final version
		assert abs(np.sum(self.U) - self.N) < self.epsilon, 'Model Didn\'t Fit Correctly'

	def __init_vars(self, Z, c, fuzziness_parameter, termination_criterion, norm_inducing_matrix):
		self.c = c
		self.N = Z.shape[1]
		self.n = Z.shape[0]
		self.m = fuzziness_parameter
		self.epsilon = termination_criterion
		# TODO: remove asserts in final version
		assert 1 < c < self.N, 'c must satisfy 1 < c < Number of samples'
		assert self.m > 1, 'fuzziness_parameter must be > 1'
		assert self.epsilon > 0, 'termination_criterion must be > 0'
		assert norm_inducing_matrix == 'identity' or 'diagonal' or 'mahalonobis', 'norm_inducing_matrix not valid'

		self.__init_A(norm_inducing_matrix, Z)
		self.__init_V(Z)
		self.__init_U(Z)

	def __init_A(self, norm_inducing_matrix, Z):
		if norm_inducing_matrix == 'identity':
			self.A = np.eye(self.n)
		elif norm_inducing_matrix == 'diagonal':
			z_var = np.zeros([self.n, 1])

			for i in range(self.n):
				tmp = Z[i, :]
				z_var[i] = 1/np.var(tmp)

			self.A = np.diagflat(np.reshape(z_var, [self.n, 1]))
			return
		elif norm_inducing_matrix == 'mahalonobis':
			z_mean = np.zeros([self.n, 1])

			for i in range(self.n):
				tmp = Z[i, :]
				z_mean[i] = np.mean(tmp)

			R = np.zeros([self.n, self.n])
			for i in range(self.N):
				zk = np.reshape(Z[:, i], [self.n, 1])
				diff = np.subtract(zk, z_mean)
				R = np.add(R, np.matmul(diff, diff.transpose()))

			R = 1/self.N * R
			self.A = np.linalg.inv(R)

	def __init_V(self, Z):
		self.V = np.zeros([self.n, self.c])
		for cluster in range(self.c):
			for feature in range(self.n):
				# TODO: maybe make this truly random
				self.V[feature][cluster] = random.uniform(np.min(Z[feature]), np.max(Z[feature]))

	def __init_U(self, Z):
		self.U = np.zeros([self.c, self.N])

		D = self.__compute_distances(Z)
		self.__update_partition_matrix(D, Z)

	def __D_squared(self, i, k, Z, V):
		zk = np.reshape(Z[:, k], [Z.shape[0], 1])
		vi = np.reshape(V[:, i], [V.shape[0], 1])

		# DEBUG here
		diff = np.subtract(zk, vi)
		ret = np.matmul(diff.transpose(), self.A)
		ret = np.matmul(ret, diff)

		return ret

	def __reached_termination(self, prev_U):
		diff = np.max(np.abs(np.subtract(self.U, prev_U)))
		if diff < self.epsilon:
			return True
		return False

	def __compute_cluster_means(self, Z):
		# DEBUG here
		for i in range(self.c):
			nom = np.zeros([self.n, 1])
			denom = 0

			for k in range(self.N):
				mu_power = pow(self.U[i][k], self.m)
				denom += mu_power
				nom = np.add(nom, mu_power * np.reshape(Z[:, k], [Z.shape[0], 1]))

			for row in range(self.V.shape[0]):
				self.V[row][i] = nom[row]/denom

	def __compute_distances(self, Z):
		D = np.zeros([self.c, self.N])

		for i in range(self.c):
			for k in range(self.N):
				D[i][k] = np.sqrt(self.__D_squared(i, k, Z, self.V))

		return D

	def __update_partition_matrix(self, D, Z):
		for k in range(self.N):
			all_distances_positive = True
			for i in range(self.c):
				if D[i][k] == 0:
					all_distances_positive = False

			if all_distances_positive:
				for i in range(self.c):
					d_ik = np.sqrt(self.__D_squared(i, k, Z, self.V))
					sum_distance = 0

					for j in range(self.c):
						d_jk = np.sqrt(self.__D_squared(j, k, Z, self.V))
						add = pow(d_ik / d_jk, 2 / (self.m - 1))
						sum_distance += add[0]

					self.U[i][k] = 1/sum_distance
			else:
				edit = []
				for i in range(self.c):
					if D[i][k] > 0:
						self.U[i][k] = 0
					else:
						edit.append(i)

				remaining = 1
				sum_added = 0
				for i in range(len(edit) - 1):
					self.U[edit[i]][k] = random.uniform(0, remaining)
					sum_added += self.U[edit[i]][k]
					remaining -= sum_added

				self.U[edit[len(edit) - 1]][k] = remaining
