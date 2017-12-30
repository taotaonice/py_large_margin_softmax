import tensorflow as tf
import numpy as np


def weight_var(shape):
    init = tf.truncated_normal(shape=shape, mean=0.001, stddev=0.01, dtype=tf.float32)
    return tf.Variable(init)


def bias_var(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=0.001, stddev=0.01))


def conv_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Net:
    def __init__(self, batch_size,  margin):
        self.init_var()
        self.img_size = 28
        self.batch_size = batch_size
        self.margin = margin

        self.test_input = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.test_label = tf.placeholder(tf.int32, [None])
        self.test_out = self.network(self.test_input)

    def init_var(self):
        self.w1 = weight_var([5, 5, 1, 32])
        self.b1 = bias_var([32])
        self.w2 = weight_var([3, 3, 32, 64])
        self.b2 = bias_var([64])
        self.w3 = weight_var([3, 3, 64, 128])
        self.b3 = bias_var([128])

        self.w4 = weight_var([4 * 4 * 128, 512])
        self.b4 = bias_var([512])
        self.w5 = weight_var([512, 2])
        self.b5 = bias_var([2])
        self.w6 =weight_var([2, 10])


    def network(self, x):
        # part 1:
        self.conv_1 = tf.nn.relu(conv_2d(x, self.w1) + self.b1)
        self.pool_1 = max_pool2x2(self.conv_1)
        self.conv_2 = tf.nn.relu(conv_2d(self.pool_1, self.w2) + self.b2)
        self.pool_2 = max_pool2x2(self.conv_2)
        self.conv_3 = tf.nn.relu(conv_2d(self.pool_2, self.w3) + self.b3)
        self.pool_3 = max_pool2x2(self.conv_3)
        self.a = tf.reshape(self.pool_3, [-1, 4 * 4 * 128])
        self.ip1 = tf.nn.relu(tf.matmul(self.a, self.w4) + self.b4)
        self.ip2 = tf.matmul(self.ip1, self.w5) + self.b5
        self.ip3 = tf.matmul(self.ip2, self.w6)
        return self.ip3

    def fi(self, theta):
        k = tf.floor(theta*self.margin/np.pi)
        return tf.pow(-1.0, k)*tf.cos(self.margin*theta)-2.0*k

    def loss(self):
        x = self.ip2 #batch_sizex2
        w = self.w6 #2x10
        xnorm = tf.sqrt(tf.reduce_sum(tf.pow(x, 2.0), 1)+1e-6) #(batch_size,)
        wnorm = tf.sqrt(tf.reduce_sum(tf.pow(w, 2.0), 0)+1e-6) #(c,)
        prod = tf.matmul(x, w) #(nxc)
        dot = tf.matmul(tf.expand_dims(xnorm, 1), tf.expand_dims(wnorm, 0)) #(nxc)
        theta = tf.acos(prod/dot)
        down = tf.reduce_sum(tf.exp(prod), 1) #nx1
        up = tf.exp(dot*self.fi(theta)) # nxc
        res = up/(tf.stack([down]*10, 1)-tf.exp(prod)+up) #nxc
        res = -tf.log(res+1e-6) #nxc
        loss = tf.zeros([1], tf.float32)
        for i in range(self.batch_size):
            loss = loss+res[i, self.test_label[i]]
        return loss/self.batch_size
