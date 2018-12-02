import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('d:/shujv/train.csv')
time_stamp = data['time']
time_value = pd.to_datetime(time_stamp,unit='s')
# print(time_value)
time_data = pd.DatetimeIndex(time_value)
data['weekday'] = time_data.weekday
data['month'] = time_data.month
data['day'] = time_data.day
data.drop(labels=['time'],axis=1)
place_count = data.groupby('place_id').count()
index_values = place_count[place_count.row_id >20].reset_index()
data = data[data['place_id'].isin(index_values.place_id)]
print(data)
data = data.query('x<1 & y<1')
x = data.drop(['place_id','time','row_id','accuracy'],axis=1,inplace=False)
y = data['place_id']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
print(knn.score(X_test,y_test))