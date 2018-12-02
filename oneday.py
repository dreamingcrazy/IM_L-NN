from sklearn.feature_extraction import DictVectorizer

data = [{'city':'杭州','wu':'一般'},{'city':'上海','wu':'无'},{'city':'北京','wu':'严重'},{'city':'成都','wu':'一般'}]
print(data)
dvt = DictVectorizer()
data_fit = dvt.fit_transform(data)
print(dvt.get_feature_names())
print(data_fit.toarray())
print(dvt.inverse_transform(data_fit))