# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a simple implementation of CGAN. When trained, the network will be able to generate an image of a user specified number using a given latent vector.


"""

import tensorflow as tf
import numpy as np
import math
import datetime
import time
import matplotlib.pyplot as plt
import pathlib

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

sample_image = mnist.train.next_batch(1)[0]

# Show a sample image of mnist
sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image, cmap = 'Greys')

# Definition of the discriminator
def discriminator(images, labels, reuse = False):
    if(reuse):
        tf.get_variable_scope().reuse_variables()
    
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer = tf.truncated_normal_initializer(stddev= 0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer = tf.constant_initializer(0))
    
    d1 = tf.nn.conv2d(input = images, filter = d_w1, strides = [1, 1, 1, 1], padding = 'SAME')
    d1 = d1 + d_b1  
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(value = d1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer = tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer = tf.constant_initializer(0))
    
    d2 = tf.nn.conv2d(input = d1, filter = d_w2, strides = [1, 1, 1, 1], padding = 'SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(value = d2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    d_w3s = tf.get_variable('d_w3s', [7 * 7 * 64 + 10, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3s = tf.get_variable('d_b3s', [1024], initializer = tf.constant_initializer(0))
    
    d3s = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3s = tf.concat([d3s, labels], 1)
    d3s = tf.matmul(d3s, d_w3s)
    d3s = d3s + d_b3s
    d3s = tf.nn.relu(d3s)
    
    d_w4s = tf.get_variable('d_w4s', [1024, 1], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b4s = tf.get_variable('d_b4s', [1], initializer = tf.constant_initializer(0))
    
    d4s = tf.matmul(d3s, d_w4s)
    d4s = d4s + d_b4s
    #d4s = tf.sigmoid(d4s)
    
    d_w3c = tf.get_variable('d_w3c', [7 * 7 * 64, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3c = tf.get_variable('d_b3c', [1024], initializer = tf.constant_initializer(0))
    
    d3c = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3c = tf.matmul(d3c, d_w3c)
    d3c = d3c + d_b3c
    d3c = tf.nn.relu(d3c)
    
    d_w4c = tf.get_variable('d_w4c', [1024, 10], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b4c = tf.get_variable('d_b4c', [10], initializer = tf.constant_initializer(0))
    
    d4c = tf.matmul(d3c, d_w4c)
    d4c = d4c + d_b4c
    d4c = tf.nn.softmax(d4c)
    
    return (d4s, d4c, labels)

# Definition fo the generator
def generator(z, c, batch_size, z_dim = 100, c_dim = 10, reuse = False):
    if(reuse):
        tf.get_variable_scope().reuse_variables()
        
    dim = z_dim + c_dim
    g_w1 = tf.get_variable('g_w1', [dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    zc = tf.concat([z, c], 1)
    g1 = tf.matmul(zc, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, math.ceil(dim/2)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [math.ceil(dim/2)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(input = g1, filter = g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, math.ceil(dim/2), math.ceil(dim/4)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [math.ceil(dim/4)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(input = g2, filter = g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, math.ceil(dim/4), 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(input = g3, filter = g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

# Starting the session
sess = tf.Session()

# Define variables
z_dimensions = 100
l_dimensions = 10
batch_size = 50

# Defining placeholders for network inputs
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding a latent vector to the generator

l_placeholder = tf.placeholder(tf.float32, [None, l_dimensions], name='l_placeholder')
# l_placeholder is for feeding groundtruth labels to the network

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator

rl_placeholder = tf.placeholder(tf.int32, shape = [None], name='rl_placeholder')
# rl_placeholder is for feeding a number label to the onehot converter

# Defining network tensors
with tf.variable_scope(tf.get_variable_scope()):
    Gz = generator(z_placeholder, l_placeholder, batch_size, z_dimensions, l_dimensions) 
    # Gz holds the generated images
    
    Dx = discriminator(x_placeholder, l_placeholder) 
    # Dx will hold discriminator prediction probabilities
    # for the real MNIST images
    
    Dg = discriminator(Gz, l_placeholder, reuse=True)
    # Dg will hold discriminator prediction probabilities for generated images
    
    DTest = discriminator(x_placeholder, l_placeholder, reuse=True)
    
    generated_images = generator(z_placeholder, l_placeholder, 1, z_dimensions, l_dimensions, True)
    
    convert_onehot = tf.one_hot(rl_placeholder, 10)

# Defining losses
lam = 0.7
d_loss_real_score = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx[0], labels=tf.ones_like(Dx[0])))
d_loss_real_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Dx[1], labels=Dx[2]))
d_loss_real = lam * d_loss_real_score + (1 - lam) * d_loss_real_class
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg[0], labels=tf.zeros_like(Dg[0])))

d_loss = d_loss_real + d_loss_fake

g_loss_score = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg[0], labels=tf.ones_like(Dg[0])))
g_loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Dg[1], labels=Dg[2]))
g_loss = lam * g_loss_score + (1 - lam) * g_loss_class

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# Defining trainers
# Train the discriminator
d_trainer = tf.train.AdamOptimizer(0.0003).minimize(d_loss, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

saver = tf.train.Saver()

#Generating Tensorboard stats
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, l_placeholder, batch_size, z_dimensions, l_dimensions, True)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_batch = mnist.train.next_batch(batch_size)
    real_image_batch = real_batch[0].reshape([batch_size, 28, 28, 1])
    real_label_batch = real_batch[1]
    real_label_batch = sess.run(convert_onehot, feed_dict={rl_placeholder: real_label_batch})
    _, dLossRealScore, dLossRealClass, dLossFake = sess.run([d_trainer, d_loss_real_score, d_loss_real_class, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch, l_placeholder: real_label_batch})

    if(i % 100 == 0):
        print("dLossRealScore:", dLossRealScore, "dLossRealClass:", dLossRealClass, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(100000):
    loop_start_t = time.perf_counter()
    real_batch = mnist.train.next_batch(batch_size)
    real_image_batch = real_batch[0].reshape([batch_size, 28, 28, 1])
    real_label_batch = real_batch[1]
    real_label_batch = sess.run(convert_onehot, feed_dict={rl_placeholder: real_label_batch})
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images with groundtruth labels
    _, dLossReal, dLossFake = sess.run([d_trainer, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch, l_placeholder: real_label_batch})

    # Train generator using a random laten vector and random label
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch, l_placeholder: real_label_batch})

    if i % 10 == 0:
        #print('Update summary')
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch, l_placeholder: real_label_batch})
        writer.add_summary(summary, i)

    if i % 100 == 0:
       # print('Testing GAN')
        # Every 100 iterations, show generated images of all numbers
        print("Iteration:", i, "at", datetime.datetime.now())
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)
        
        f, axarr = plt.subplots(2, 5, figsize=(10, 4))
        for a in range(2):
            for b in range(5):
                axarr[a, b].axis('off')
        
        # Generate a random latent vector        
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        for a in range(2):
            for b in range(5):
                n = a * 5 + b
                labels = np.zeros((1, l_dimensions))
                labels[0][n] = 1.0
                print('Generate Class: ', np.argmax(labels, axis = 1))
                
                # Generate an image of number n
                images = sess.run(generated_images, {z_placeholder: z_batch, l_placeholder: labels})
                axarr[a, b].imshow(images[0].reshape([28, 28]), cmap='Greys')
                
                # Show discriminator's estimate of the generated image
                im = images[0].reshape([1, 28, 28, 1])
                estimate = sess.run(Dx, {x_placeholder: im, l_placeholder: labels})
                print("Estimate Score:", estimate[0])
                print("Estimate Class:", np.argmax(estimate[1]))
                
                pathlib.Path('results/{}'.format(n)).mkdir(parents=True, exist_ok=True)
                file_name = 'results/{}/{}_targ_{}_est_{}_score_{}.png'.format(n, i, n, np.argmax(estimate[1]), estimate[0][0][0])
                
                # Save generated images individualy
                extent = axarr[a, b].get_window_extent().transformed(f.dpi_scale_trans.inverted())
                # Pad the saved area by 10% in the x-direction and 20% in the y-direction
                f.savefig(file_name, bbox_inches=extent.expanded(1.1, 1.2))
                
        pathlib.Path('results/all').mkdir(parents=True, exist_ok=True)
        file_name = 'results/all/{}_results.png'.format(i)
        
        # Save the combined image of all generated images
        plt.savefig(file_name)
        plt.show()
        
        # Test the discrimintor
        test_image = mnist.train.next_batch(1)
        test_label = np.zeros((1, l_dimensions))
        test_label[0][test_image[1]] = 1.0
        test_image = test_image[0].reshape([1, 28, 28, 1])
        
        dscore, dclass, _ = sess.run(DTest, {x_placeholder: test_image, l_placeholder: test_label})
        
        plt.imshow(test_image.reshape([28, 28]), cmap='Greys')
        plt.show()
        
        print('Score: ', dscore)
        print('Class', np.argmax(dclass))
    
    loop_end_t = time.perf_counter()
    
    #print('Loop ', i, ',  Loop time: ', loop_end_t - loop_start_t, ' s')

saver.save(sess, './model_cgan.ckpt')

sess.close()