from sklearn.decomposition import PCA
data = [[1,2,3,4],[4,5,6,7],[7,8,9,10]]

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
print(data)
print(data_pca)