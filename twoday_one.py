from sklearn.feature_extraction.text import CountVectorizer
str1 = 'i like python , python is very good.'
str2 = 'python is number one'

str3 = '我 的 家 在 东北，我 爱吃 林蛙！'
str4 = '我是东北人，我不喜欢别人瞅我！'


cvt = CountVectorizer(stop_words=['is','very'])
data = cvt.fit_transform([str1,str2,str3,str4])
# 单个英文不能进入词频统计
print(cvt.get_feature_names())
print(data.toarray())