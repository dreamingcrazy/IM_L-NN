# 归一化

from sklearn.preprocessing import MinMaxScaler

# [[160 50 2]
# [180 1  0]
# [189 0  0]]

mms = MinMaxScaler()


data = [[160, 50, 2],[180 ,1 , 0],[189 ,0  ,0]]

print(data)

print(mms.fit_transform(data))
