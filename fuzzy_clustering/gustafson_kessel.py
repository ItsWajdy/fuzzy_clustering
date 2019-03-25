from fuzzy_clustering import c_means
import numpy as np


class Model(c_means.Model):
	def __init__(self):
		self.F = []
		self.rho = None
		super(Model, self).__init__()

	def __init_vars(self, Z, c, fuzziness_parameter, termination_criterion, norm_inducing_matrix):
		self.rho = np.ones(c)
		super(Model, self).__init_vars(Z, c, fuzziness_parameter, termination_criterion, norm_inducing_matrix)

	# TODO: make this initialize U with random values and not dependant on parent class at all
	def __init_U(self, Z):
		self.U = np.zeros([self.c, self.N])

		D = super(Model, self).__compute_distances(Z)
		self.__update_partition_matrix(D, Z)

	def __compute_distances(self, Z):
		self.__compute_covariance_matrices(Z)
		return super(Model, self).__compute_distances(Z)

	def __D_squared(self, i, k, Z, V):
		zk = np.reshape(Z[:, k], [Z.shape[0], 1])
		vi = np.reshape(V[:, i], [V.shape[0], 1])

		A = pow(self.rho[i] * np.linalg.det(self.F[i]), 1/self.n) * np.linalg.inv(self.F[i])

		# DEBUG here
		diff = np.subtract(zk, vi)
		ret = np.matmul(diff.transpose(), A)
		ret = np.matmul(ret, diff)

		return ret

	def __compute_covariance_matrices(self, Z):
		for i in range(self.c):
			mu_power = 0
			vi = np.reshape(self.V[:, i], [self.V.shape[0], 1])

			f = np.zeros([self.n, self.n])
			for k in range(self.N):
				mu_power += pow(self.U[i][k], self.m)

				zk = np.reshape(Z[:, k], [Z.shape[0], 1])
				diff = np.subtract(zk, vi)
				# DEBUG HERE
				add = pow(self.U[i][k], self.m) * np.matmul(diff, diff.transpose())
				f = np.add(f, add)

			self.F.append(f/mu_power)
