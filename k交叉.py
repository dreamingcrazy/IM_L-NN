from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
iris = load_iris()
data = iris.data
target = iris.target
print(cross_val_score(kn,data,target,cv=10))