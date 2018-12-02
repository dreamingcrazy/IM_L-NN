from sklearn.tree import DecisionTreeClassifier,export_graphviz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pydotplus

dtc = DecisionTreeClassifier( criterion='entropy')

data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(data)

x= data[['pclass','age','sex']]
x['age'].fillna(x['age'].mean(),inplace=True)
# print(x)
y = data.survived
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train = x_train.to_dict(orient='records')
# 5、当关键字orient=’records’ 时
# 形成[{column -> value}, … , {column -> value}]的结构
# 整体构成一个列表，内层是将原始数据的每行提取出来形成字典
x_test = x_test.to_dict(orient='records')
# print(x_train)

dvt = DictVectorizer(sparse=False)

x_train=dvt.fit_transform(x_train)
x_test=dvt.transform(x_test)
# print(dvt.get_feature_names())

dtc.fit(x_train,y_train)
y_predict = dtc.predict(x_test)
print('准确率',dtc.score(x_test,y_test))

