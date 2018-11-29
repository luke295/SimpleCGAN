# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a simple implementation of CGAN. When trained, the network will be able to generate an image of a user specified number using a given latent vector.


"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Definition of the discriminator
def discriminator(images, labels, reuse = False):
    if(reuse):
        tf.get_variable_scope().reuse_variables()
    
    d_w1 = tf.get_variable('d_w1', [5, 5, 3, 32], initializer = tf.truncated_normal_initializer(stddev= 0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer = tf.constant_initializer(0))
    
    d1 = tf.nn.conv2d(input = images, filter = d_w1, strides = [1, 1, 1, 1], padding = 'SAME')
    d1 = d1 + d_b1  
    d1 = tf.nn.leaky_relu(d1)
    d1 = tf.nn.avg_pool(value = d1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer = tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer = tf.constant_initializer(0))
    
    d2 = tf.nn.conv2d(input = d1, filter = d_w2, strides = [1, 1, 1, 1], padding = 'SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.leaky_relu(d2)
    d2 = tf.nn.avg_pool(value = d2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    d_w3s = tf.get_variable('d_w3s', [8 * 8 * 64 + 10, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3s = tf.get_variable('d_b3s', [1024], initializer = tf.constant_initializer(0))
    
    d3s = tf.reshape(d2, [-1, 8 * 8 * 64])
    d3s = tf.concat([d3s, labels], 1)
    d3s = tf.matmul(d3s, d_w3s)
    d3s = d3s + d_b3s
    d3s = tf.nn.leaky_relu(d3s)
    
    d_w4s = tf.get_variable('d_w4s', [1024, 1], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b4s = tf.get_variable('d_b4s', [1], initializer = tf.constant_initializer(0))
    
    d4s = tf.matmul(d3s, d_w4s)
    d4s = d4s + d_b4s
    #d4s = tf.sigmoid(d4s)
    
    d_w3c = tf.get_variable('d_w3c', [8 * 8 * 64, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3c = tf.get_variable('d_b3c', [1024], initializer = tf.constant_initializer(0))
    
    d3c = tf.reshape(d2, [-1, 8 * 8 * 64])
    d3c = tf.matmul(d3c, d_w3c)
    d3c = d3c + d_b3c
    d3c = tf.nn.leaky_relu(d3c)
    
    d_w4c = tf.get_variable('d_w4c', [1024, 10], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b4c = tf.get_variable('d_b4c', [10], initializer = tf.constant_initializer(0))
    
    d4c = tf.matmul(d3c, d_w4c)
    d4c = d4c + d_b4c
    #d4c = tf.nn.softmax(d4c)
    
    return (d4s, d4c, labels)

# Definition fo the generator
def generator(z, c, batch_size, z_dim = 100, c_dim = 10, reuse = False):
    if(reuse):
        tf.get_variable_scope().reuse_variables()
        
    dim = z_dim + c_dim
    g_w1 = tf.get_variable('g_w1', [dim, 12288], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [12288], initializer=tf.truncated_normal_initializer(stddev=0.02))
    zc = tf.concat([z, c], 1)
    g1 = tf.matmul(zc, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 64, 64, 3])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 3, math.ceil(dim/2)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [math.ceil(dim/2)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(input = g1, filter = g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [64, 64])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, math.ceil(dim/2), math.ceil(dim/4)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [math.ceil(dim/4)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(input = g2, filter = g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [64, 64])

    # Final convolution with three output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, math.ceil(dim/4), 3], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [3], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(input = g3, filter = g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

sess = tf.Session()
z_dimensions = 100
l_dimensions = 10
batch_size = 1

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding a given laten vector to the generator

l_placeholder = tf.placeholder(tf.float32, [None, l_dimensions], name='l_placeholder')
# l_placeholder is for feeding groundtruth labels to the network

x_placeholder = tf.placeholder(tf.float32, shape = [None,32,32,3], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator

rl_placeholder = tf.placeholder(tf.int32, shape = [None], name='rl_placeholder')
# rl_placeholder is for feeding a number label to the onehot converter

with tf.variable_scope(tf.get_variable_scope()):
    Gz = generator(z_placeholder, l_placeholder, batch_size, z_dimensions, l_dimensions) 
    # Gz holds the generated images
    
saver = tf.train.Saver()
saver.restore(sess, "./model_cgan.ckpt")

while(input('Generate new?') == 'yes'):
    f, axarr = plt.subplots(2, 5)
    z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
    d_time = np.zeros(10)
    for a in range(2):
        for b in range(5):
            labels = np.zeros((1, l_dimensions))
            labels[0][a * 5 + b] = 1.0
            s_time = time.perf_counter()
            images = sess.run(Gz, {z_placeholder: z_batch, l_placeholder: labels})
            e_time = time.perf_counter()
            d_time[a * 5 + b] = e_time - s_time
            axarr[a, b].imshow(images[0].reshape([32, 32, 3]))
            axarr[a, b].axis('off')           
    plt.show()
    print('Average generation time:', np.average(d_time) * 1000, 'ms')

sess.close()