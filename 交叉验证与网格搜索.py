from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
data = iris.data
target = iris.target
# print(data)
# print(target)

# 将原始数据集分为测试集与训练集
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2)
kn = KNeighborsClassifier()


cv_pa = GridSearchCV(kn,param_grid={'n_neighbors':[1,3,5,7,9]},cv=10)
print('cv',cv_pa)
cv_pa.fit(x_train,y_train)
score = cv_pa.score(x_test,y_test)
print('这是使用改模型得到的最好的准确率',score)
print('这是最好的模型',cv_pa.best_estimator_)
print('最好的参数',cv_pa.best_params_)
print('这是最好的结果',cv_pa.cv_results_)