import tensorflow as tf

# 权重
def weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape))
# 偏置值

def baiss(shape):
    return tf.Variable(initial_value=tf.constant(0.0,shape=shape))


def model():
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,shape=[None,28*28*1])
        y = tf.placeholder(tf.float32,shape=[None,10])



    with tf.variable_scope('layer1'):
        # 卷积
        # 设置filter,首先是权重值，然后知偏置值


        weight = weights(shape=(5,5,1,32))
        bais = baiss(shape=[32])

        x_reshape = tf.reshape(x,shape=[-1,28,28,1])
        # 卷积完成之后输出的数据形态 [-1，28,28,32]
        conv1 = tf.nn.conv2d(input=x_reshape,filter=weight,strides=[1,1,1,1],padding='SAME') + bais
        '''input, filter, strides, padding, use_cudnn_on_gpu=None,
               data_format=None, name=None'''
        # 激活
        x_relu = tf.nn.relu(conv1)
        # 池化 输出形状[-1,14,14,32]
        x_pool1=tf.nn.max_pool(value=x_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        '''[ -1  ,   10] = [-1,14*14*32 ]      [ 14*14*32   ,10]
        [-1,10] = x_pool1 * weight + [10]
        y_predict = x_pool1 * weight + bais '''