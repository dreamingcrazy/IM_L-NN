from sklearn.datasets import load_iris,fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
fetch_20newsgroups()
data = iris.data
target = iris.target
print(data)
print(target)

# 将原始数据集分为测试集与训练集
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2)
#  选择K近邻算法的API接口
kn = KNeighborsClassifier()


# 训练数据，完成模型的训练
kn.fit(x_train,y_train)

# 进行测试集的预测
predict_y = kn.predict(x_test)
print('这是预测值',predict_y)
print('真实的结果',y_test)

sc = kn.score(x_test,y_test)
print('这是预测结果的准确率',sc)