import pandas as pd
import numpy as np

df = pd.DataFrame({'key1':['a','a','b','b','a','a'],'key2':['one','two','one','two','one','one'],'data1':[1,2,3,4,5,1],'data2':[6,7,8,9,10,1]})
print(df)
group1 = df.groupby('key1')
# for i in group1:
#     print(i)
# print('group1',group1.size())

group2 = df.groupby(['key1','key2'])
# for j in group2:
#     print(j)

# print('group2',group2.size())

# print(pd.pivot_table(df, index='key1', columns='key2', aggfunc={'data1':sum},margins=True,values=['data1']))
print(pd.crosstab(df.key1,df.key2, margins=True))
#[Out]# key2  one  two  All
#[Out]# key1
#[Out]# a       2    1    3
#[Out]# b       1    1    2
#[Out]# All     3    2    5
