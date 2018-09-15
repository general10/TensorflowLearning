from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 下载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 注册 session
sess = tf.InteractiveSession()

# 创建数据集Placeholder
# None表示不限条数 784表示输入是784维的向量
x = tf.placeholder(tf.float32, [None, 784])


# 下面是公式 P49
# w是输入数据
# 784代表特征的维数
# 10代表结果 因为是识别手写数字 所以一共10维代表0-9
# b代表bias 数据本身的偏向
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

# y_是真实概率分布
# cross_entropy是信息熵 数学之美里有介绍 在这里作为损失函数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化
tf.global_variables_initializer().run()

# 每次取1000个样本进行训练
# 书上是取100训出来的结果是92.05%(虽然每次也都在变 但变化不大)
# 改成1000之后训出来是92.57% 表示根本没啥差距= =
# 改成10000也没多多少 所以还有要往后学更好的方法(函数)2333
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10000)
    train_step.run({x: batch_xs, y_: batch_ys})

# 最后返回计算分类是否正确的操作 correct_prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 把correct_prediction转成float32(double不行吗真是的= =) 再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 输出准确率
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
