from sklearn.datasets import load_iris
from fuzzy_clustering.c_means import Model as C
from fuzzy_clustering.gustafson_kessel import Model as GK


iris = load_iris()
data = iris.data
labels = iris.target

Z = data.transpose()

c_means = C()
gk = GK()

c_means.fit(Z, 3)
gk.fit(Z, 3)
