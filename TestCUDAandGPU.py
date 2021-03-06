# 安装教程 https://blog.csdn.net/weixin_42359147/article/details/80622306

import tensorflow as tf

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用

b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)                                  # 判断GPU是否可以用

print(a)
print(b)

#Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
#Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#Runs the op.
print(sess.run(c))