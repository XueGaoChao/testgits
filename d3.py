#-*-coding:utf-8-*-
# from __future__ import print_function
# from __future__ import division
# from __future__ import absolute_import
# 导入tensorflow库
import tensorflow as tf
# 调用手写体的四个数据包，训练集、训练标签集、测试集、测试标签集
from tensorflow.examples.tutorials.mnist import input_data

# 设置tensorboard日志存储路径
LOGDIR = 'D:\log'
# 定义卷积核函数，（输入图像，输入通道，输出通道，命名空间）
def conv_layer(input, size_in,size_out, name='conv'):
    # 命名空间，tf.get_variable不受name_scope(name)的影响已经声明的变量无需再声明
    with tf.name_scope(name):
        # 定义变量 截断到随机变量，标准差
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out],stddev=0.1),name='w')
        # 定义常量张量
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='b')
        # 进行卷积，（输入图像数，卷积核，步长，填充）
        conv = tf.nn.conv2d(input, w,strides=[1, 1, 1, 1],padding='SAME')
        # 定义模型
        act = tf.nn.relu(conv + b)
        # 以直方图的形式在tensorboard上展示
        tf.summary.histogram('w',w)
        tf.summary.histogram('b',b)
        tf.summary.histogram('act',act)
        return act

# 定义全连接层函数，（输入图像，输入通道，输出通道，命名）
def fc_layer(input, size_in,size_out, name='fc'):
    # 定义命名空间
    with tf.name_scope(name):
        # 截断随机变量
        w = tf.Variable(tf.truncated_normal([size_in, size_out],stddev=0.1),name='w')
        # 以list或数据的形式输入到常量张量
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='b')
        # 定义模型
        act = tf.matmul(input, w) + b
        # 以直方图的形式在tengsorboard上展示
        tf.summary.histogram('w',w)
        tf.summary.histogram('b',b)
        tf.summary.histogram('act',act)
        return act

# 定义数据集函数
def mnist():
    # 启动sess会话
    sess = tf.Session()
    # 命名空间‘input
    with tf.name_scope('input'):
        # 定义占位符x,shape=[None, 784]
        x = tf.placeholder(tf.float32, [None, 784])
        # 保留原数据将x以28*28的维度输出
        x_img = tf.reshape(x, [-1, 28, 28, 1])
        # 定义占位符y,shape=[None, 10]
        y = tf.placeholder(tf.float32, [None, 10])
        #
        tf.summary.image('image',x_img,max_outputs=3)
    # 调用卷积函数，进行传参
    conv1 = conv_layer(x_img,1, 32, name='conv1')
    # 对数据进行最大池化
    conv1_pool = tf.nn.max_pool(conv1,ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],padding='SAME')
    conv2 = conv_layer(conv1_pool, 32, 64, name='conv2')
    conv2_out = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],padding='SAME')
    # 将输出图像维度设置为7*7*64
    flatened = tf.reshape(conv2_out, [-1, 7 * 7 * 64])

    # 使用激活函数将数据传入全连接函数
    fc1 = tf.nn.relu(fc_layer(flatened, 7 * 7 * 64,1024,name='fc1'))

    # 未归一化的概率
    logits = fc_layer(fc1, 1024, 10 ,name='logits')

    # 命名空间'cross_entropy'
    with tf.name_scope('cross_entropy'):
        # 计算代价函数
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=y),name='logits')
        # 在tengsorboard上画出loss曲线
        tf.summary.scalar('loss',xent)
    # 命名空间‘accuracy'
    with tf.name_scope('accuracy'):
        # 预测的数据准确率
        correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
        # 将准确率转化为float类型
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        # 输出accuracy
        tf.summary.scalar('accuracy',accuracy)
    # 命名空间’train'
    with tf.name_scope('train'):
        # 优化器
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)
    # 调用MNIST_data的四个数据集返回mnist对象，进行独热编码
    mnist = input_data.read_data_sets(train_dir='MNIST_data',one_hot=True)

    # 对全局变量进行初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 写入tensorboard日志文件
    file_write = tf.summary.FileWriter(LOGDIR)
    file_write.add_graph(sess.graph)
    # 自动管理
    merge_op = tf.summary.merge_all()
    # 开始学习训练
    for i in range(1001):
        # 取100个作为训练集
        batch = mnist.train.next_batch(100)
        # i等于5的时候将summaries输入到tensorboard日志中
        if i % 5 == 0:
            [summaries,_] = sess.run([merge_op,accuracy],feed_dict={
                x:batch[0],y:batch[1]})
            file_write.add_summary(summaries, i)
        # i 等于100的时候输出step步长和accuracy准确率
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                    x:batch[0],y:batch[1]})
            print('step: %d accuracy: %g' % (i, train_accuracy))
        sess.run(train_step,feed_dict={x:batch[0],y:batch[1]})

# 在主函数调用
if __name__ == '__main__':
    mnist()