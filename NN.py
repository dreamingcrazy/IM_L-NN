import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./data/',one_hot=True)
print(mnist)

feature = mnist.train.images

label = mnist.train.labels

print(feature)
print(label)


#数据
with tf.variable_scope('data'):
    x = tf.placeholder(dtype=tf.float32,shape=[None,784])
    y = tf.placeholder(dtype=tf.float32,shape=[None,10])

#模型
with tf.variable_scope('model'):


    #矩阵相乘   [none,784]*[784,10]               =[None,10]
    weight = tf.Variable(initial_value=tf.random_normal(shape=(784,10),dtype=tf.float32),name='weight')
    bias = tf.Variable(initial_value=0.0,name='bias')

#预测
    y_predict = tf.matmul(x,weight) + bias

# 计算损失

with tf.variable_scope('softmax-cross'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y)
#计算平均损失
    mean_loss = tf.reduce_mean(loss)
# 梯度下降优化损失

with tf.variable_scope('sgd'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(mean_loss)

#计算准确率

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(1000):
        x_train,y_train = mnist.train.next_batch(100)
        sess.run(train_op,feed_dict={x:x_train,y:y_train})
