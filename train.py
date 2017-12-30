import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import Net

mnist = np.load('mnist_data_label.npz')
image = (mnist['data']/255.0-0.5)*2
label = mnist['label']

test = np.load('test_data_label.npz')
test_img = (test['data']/255.0-0.5)*2
test_label = test['label']


batch_size = 64
max_iter = 100000
margin = 2

net = Net.Net(batch_size, margin)
loss = net.loss()
train_step = tf.train.AdamOptimizer(0.00003).minimize(loss)

np.random.seed(1997)
rdp = np.random.randint(0, len(label), [max_iter, batch_size])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

plt.figure(0)
plt.ion()

for i in range(max_iter):
    img = image[rdp[i, :], :, :]
    img = np.expand_dims(img, 3)
    lb = label[rdp[i, :]]
    sess.run(train_step, feed_dict={net.test_input: img, net.test_label: lb})
    if i % 100 == 0:
        [l] = sess.run(loss, feed_dict={net.test_input: img, net.test_label: lb})
        print 'iter:', i, 'loss:', l
        plt.plot(i, l, 'rx')
        plt.pause(0.00001)

saver = tf.train.Saver()
saver.save(sess, 'l_softmax.ckpt')

# test
plt.figure(1)
plt.ion()

for i in range(5000):
    img = test_img[i, :, :]
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    lb = test_label[i]
    feature = sess.run(net.ip2, feed_dict={net.test_input: img})
    feature = feature[0]
    color = lb*236876
    color = '#%06x'%color
    plt.plot(feature[0], feature[1], '.', color=color)
    plt.pause(0.00001)

plt.ioff()
plt.show()
sess.close()
