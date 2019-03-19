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
			Z: np.ndarray,
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

		self.c = c
		self.N = Z.shape[1]
		self.n = Z.shape[0]
		self.m = fuzziness_parameter
		self.epsilon = termination_criterion
		assert 1 < c < self.N, 'c must satisfy 1 < c < Number of samples'
		assert self.m > 1, 'fuzziness_parameter must be > 1'
		assert self.epsilon > 0, 'termination_criterion must be > 0'
		assert norm_inducing_matrix == 'identity' or 'diagonal' or 'mahalonobis', 'norm_inducing_matrix not valid'

		self.__init_A(norm_inducing_matrix)
		self.__init_V(Z)
		self.__init_U(Z)

	def __init_A(self, norm_inducing_matrix):
		if norm_inducing_matrix == 'identity':
			self.A = np.eye(self.n)
		elif norm_inducing_matrix == 'diagonal':
			# TODO: implement different types of A
			pass
		elif norm_inducing_matrix == 'mahalonobis':
			pass

	def __init_V(self, Z):
		self.V = np.zeros([self.n, self.c])
		for cluster in range(self.c):
			for feature in range(self.n):
				# TODO: maybe make this truly random
				self.V[feature][cluster] = random.uniform(np.min(Z[feature]), np.max(Z[feature]))

	def __init_U(self, Z):
		self.U = np.zeros([self.c, self.N])

		for cluster in range(self.c):
			for sample in range(self.N):
				d_ik = np.sqrt(self.__D_squared(cluster, sample, Z, self.V))
				sum_distance = 0

				for j in range(self.c):
					d_jk = np.sqrt(self.__D_squared(j, sample, Z, self.V))
					add = pow(d_ik/d_jk, 2/(self.m-1))
					sum_distance += add[0]

				self.U[cluster][sample] = sum_distance

	def __D_squared(self, i, k, Z, V):
		zk = Z[:, k]
		vi = V[:, i]

		diff = np.subtract(zk, vi)
		ret = np.matmul(diff, self.A)
		ret = np.multiply(ret, diff)

		return ret
