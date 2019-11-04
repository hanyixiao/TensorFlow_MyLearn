#Tensorflow训练神经网络
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
#载入mnist的数据集
#MINIST数据集相关的常数
INPUT_NODE = 784     #输入层的节点数　  图片的像素
OUTPUT_NODE = 10     #输出层的节点数    类别的个数

#配置神经网络的参数
LAYER1_NODE = 500    #隐藏层的节点数

BATCH_SIZE = 100     #一个训练batch中的数据个数

LEARNING_RATE_BASE = 0.8   #基础的学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率

REGULARIZATION_RATE = 0.0001    #描述模型复杂程度的正则化项在损失函数的系数
TRAINING_STEPS = 30000          #训练轮数
MOVING_AVERAGE_DECAY = 0.99     #滑动平均的衰减率
#一个辅助函数，给定圣经网络的输入和输出和所有参数，计算神经网络的前向传输结果
def inference(input_tensor,avg_class,weights1,biasses1,
            weights2,biasses2):
            #当没有提供滑动平均类的时候，直接使用参数的当前的取值
            if avg_class == None:
                #计算隐藏层的前向传播的结果，这里使用了ReLu激活函数
                layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)
                                    +biasses1)
                #计算输出层的前向传播结果
                return tf.matmul(layer1,weights2)+biasses2
            else:
                layer1 = tf.nn.relu(
                    tf.matmul(input_tensor,avg_class.average(weights1))+
                    avg_class.average(biasses1)
                )
                return (
                        tf.matmul(layer1,avg_class.average(weights2)) +
                        avg_class.average(biasses2)
                )

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = 'y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1)
    )
    biasses1 = tf.Variable(
        tf.constant(0.1,shape = [LAYER1_NODE])
    )
    #生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1)
    )
    biasses2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    #计算当前参数下的神经网络前向传播的结果，
    #函数不会使用参数的滑动平均值
    y = inference(x,None,weights1,biasses1,weights2,biasses2)
    
    #定义存储训练轮数的变量
    global_step = tf.Variable(0,trainable = False)

    #给定滑动平均的衰减率和训练轮数的变量
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,global_step
    )
    #在所有网络的变量上使用滑动平均，其他辅助变量
    variables_average_op = variable_averages.apply(
        tf.trainable_variables()
    )

    #计算使用了滑动平均之后的前向传播的结果
    average_y = inference(
        x,variable_averages,weights1,biasses1,weights2,biasses2
    )

    #计算交叉商作为刻画预测值与真实值的差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,labels = tf.argmax(y_,1)
    )
    #计算当前的batch中所有样品的交叉商的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算Ｌ2的正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算模型的正则化损失函数
    regularization = regularizer(weights1)+regularizer(weights2)
    
    #总损失等于交叉商的损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,#基础学习率，随着迭代的进行，更新变量时候使用
        global_step,       #当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,#过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY#学习率的衰减速度
    )
    #使用tf.train.GradientDescentOptiontimizer优化算法来优化损失函数。这里的损失函数包含了交叉商和正则化
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss,global_step=global_step)

    #在训练神经网络算法时，每过一遍数据既需要通过反向传播算法来更新圣经网络中的参数
    #又需要更新每一个参数的滑动平均值。为了一次万事成多个操作，TensorFlow
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name = 'train')
    #检验使用了滑动平均的模型的神经网络向前传播结果是否正确。
    #计算每一个样例的预测答案。其中average_y是一个batch_size *10的二维数组
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    #这个运算首先将一个布尔型的数值转换成实数型然后计算成功率，这个平均值就是模型在一组数据上的成功率
    accruracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #准备验证数据，一般在神经网络训练过程中会通过验证数据来大致判断停止的条件
        #和评判训练的效果
        validate_feed ={
            x: mnist.validation.images,
            y_: mnist.validation.labels 
        }
        #准备测试数据
        test_feed = {x: mnist.test.images,y_: mnist.test.labels}

        #迭代地训练神经网络
        start=time.time();
        for i in range(TRAINING_STEPS):
           
            #每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                #计算滑动平均模型在衍生数据上的结果。因为MNIST数据集比较小，所以一次
                #可以处理所有的验证数据，为了计算方便，本样例的的程序没有将验证数据划分得跟小的batch
                #太大的batch会导致计算时间过长甚至发生内存溢出烦人错误
                validate_acc = sess.run(accruracy,feed_dict = validate_feed)
                end=time.time();
                print("After %d trainning step(s),validation accuracy "
                "using average model is %g " %(i,validate_acc)
                )
                print('used time is {0:2.2f}'.format(end-start));
                start=end;
            #产生这一轮使用的一个batch的训练数据集，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict = {x: xs, y_: ys})
        #训练之后，在测试数据集桑检测网络模型的最终正确率
        test_acc = sess.run(accruracy,feed_dict = test_feed)
        print("After %d trainning step(s),test accuracy "
                "using average model is %g " %(TRAINING_STEPS,test_acc)
            )

#主程序接口
def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data",one_hot=True)
    train(mnist)

#TensorFlow提供一个主程序入口,tf,app,run会调用上面的定义的ｍｉａｎ函数
if __name__ == '__main__':
    tf.app.run()