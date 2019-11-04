from tensorflow.examples.tutorials.mnist import input_data
#载入mnist的数据集
mnist = input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)
#打印Training data size
print ("Training data size:", mnist.train.num_examples)
#打印Validating data size 
print ("Validating data size: ", mnist.validation.num_examples)
#打印Testing data size:
print ("Testing data size: ", mnist.test.num_examples)
#打印Example trainning date
print ("Examples training data: ", mnist.train.images[0])
#打印Example training data label
print ("Examples training label: ", mnist.train.labels[0])
batch_size=100
xs,ys=mnist.train.next_batch(batch_size)
#
print ("X shape:",xs.shape)
print ("Y shape:",ys.shape)
print ("#############EOF#####################")
