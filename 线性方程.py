from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def price_predict():
    '''这是用线性模型进行预测'''
    bos = load_boston()
    x = bos.data
    y = bos.target
    # print(x[1])
    # print(len(target))

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    y_predict = lr.predict(x_test)
    score = lr.score(x_test,y_test)
    print('这是真实的房价',y_test)
    print('这是预测的房价',y_predict)
#     误差看均方误差
    mean_err = mean_squared_error(y_test,y_predict)
    print('均方误差',mean_err)
    print(lr.coef_)
#     查看特征的权重系数
if __name__ == '__main__':
    price_predict()