import tensorflow as tf
from tensorflow.examples.tutorials.mnist  import input_data

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




    with tf.variable_scope('quan'):
        x_t = tf.reshape(x_pool1,shape=(-1,14*14*32))
        w = weights(shape=(14*14*32,10))
        b = baiss(shape=[10])
        y_predict =  tf.matmul(x_t,w) + b



    return x,y,y_predict
        #           [ -1  ,   10] = [-1,14*14*32 ]      [ 14*14*32   ,10]
        # [-1,10] = x_pool1 * weight + [10]
        # y_predict = x_pool1 * weight + bais

def loss_cop(y,y_predict):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict)
    mean_loss = tf.reduce_mean(loss)
    return mean_loss

def sgd(meanloss,y,y_predict):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(meanloss)
    equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
    ac = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    return train_op,ac

def main():
    minist = input_data.read_data_sets('./data',one_hot=True)
    x,y,y_predict = model()
    loss = loss_cop(y,y_predict)

    train_op,ac = sgd(loss,y,y_predict)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            x_train,y_train = minist.train.next_batch(100)
            sess.run(train_op,feed_dict={x:x_train,y:y_train})
            ret = sess.run(ac,feed_dict={x:x_train,y:y_train})
            print(ret)

if __name__ == '__main__':
    main()