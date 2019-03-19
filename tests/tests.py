from fuzzy_clustering.c_means import Model
import numpy as np


model = Model()
model.fit(np.eye(3), 2, 2, 1)

print(model.A)
print(model.V)
print(model.U)

# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(np.min(arr[0]))
