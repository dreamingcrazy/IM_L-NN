import pandas as pd

aisles = pd.read_csv('./instacart/aisles.csv') #商品与所属的具体物品的类别
order_products_prior = pd.read_csv('./instacart/order_products_prior.csv') #订单与商品信息
orders = pd.read_csv('./instacart/orders.csv') #用户的订单信息
products = pd.read_csv('./instacart/products.csv') #商品信息

# asis--products
data = pd.merge(orders,order_products_prior,on=['order_id','order_id'])
data = pd.merge(data,products,on=['product_id','product_id'])

data = pd.merge(data,aisles,on=['aisle_id','aisle_id'])
#
# print(data)
# 现在的功能，统计用户与产品类型的关系

data_after = pd.crosstab(data['user_id'],aisles['aisle'])
print(data_after)



