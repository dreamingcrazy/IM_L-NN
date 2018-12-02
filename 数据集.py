from sklearn.datasets import load_iris,fetch_20newsgroups
from sklearn.model_selection import train_test_split

iris = load_iris()
# print(iris)
data_x = iris.data
label_y = iris.target

new = fetch_20newsgroups()
# print(new.data)
# print(new.target)
# 分训练集与测试集

X_train, X_test, y_train, y_test =train_test_split(data_x,label_y)



# 题目求的是字母位置，也就是 (下标 + 1)， 所以我们计算分组的下标和字母下标就好
# 开始需要计算 组的初始下标和字母在组中的初始下标，也就是字母的  (初始位置-1)，
# """
#
# letterList = ['ABCDEFGHI', 'JKLMNOPQR', 'STUVWXYZ*']
# letterSubDict = {}
#
# def comp_sub():
#     for i, j in enumerate(letterList):
#         for a, b in enumerate(j):
#             letterSubDict[b] = [i, a]
# def encry():
#     date = input('输入日期:')
#     try:
#         dateList = [int(i) for i in date.split(' ')]
#     except Exception:
#         print("日期输入错误")
#         return
#     strInfor = input('输入字符串:')
#     strInfor = strInfor.replace(' ', '*')
#     Mnums = (dateList[0] - 1) % 3
#     Dnums = (dateList[1] - 1) % 9
#     for i, j in letterSubDict.items():
#         j[0] = str((3 + (j[0] - Mnums)) % 3 + 1)
#         j[1] = str((9 + (j[1] - Dnums)) % 9 + 1)
#         letterSubDict[i] = j
#     position = []
#     try:
#         for i in strInfor:
#             posit = ''.join(letterSubDict[i])
#             position.append(posit)
#     except Exception:
#         print("输入字符错误")
#         return
#     position = ' '.join(position)
#     print(position)
#
#
# if __name__ == '__main__':
#     comp_sub()
#     encry()
# '''此题稍难，'''
# def budget(all_money, price):
#     price_list = price.split(' ')
#     all_money = int(all_money)
#     total = sum([float(p) for p in price_list])
#     if total > all_money:
#         return '超出限制'
#     else:
#         return '放心买'
# def main():
#     all_money = input('all_money:')
#     price = input('price')
#     try:
#         result = budget(all_money, price)
#     except Exception :
#         print('输入有误')
#         return
#     print(result)
# if __name__ == '__main__':
#     main()