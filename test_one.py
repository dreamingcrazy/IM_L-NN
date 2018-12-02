from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

cols = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=cols)
# print(data)
data.replace(to_replace='?',value=np.nan,inplace=True)
data.dropna(inplace=True)

data_train = data[:600]
data_test = data[600:]

# 通过聚类完成数据的分类
kms = KMeans(n_clusters=2)

kms.fit(data_train)
print(kms.predict(data_test))
plt.figure()
