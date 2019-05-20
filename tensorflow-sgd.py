#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:55:12 2019

@author: Ling
"""

import tensorflow as tf
import numpy
import pandas as pd
import time
import csv

learning_rate = 0.001
maxEpochs = 1000
verbose = 100
dimension = 100
n = 1000

train_X = numpy.zeros(shape=(dimension, n))
train_Y = numpy.zeros(n)

for i in range(1,101):
    file_name = 'data/xcol'+ str(i) + '.csv'
    train_X[i-1] = numpy.asarray(pd.read_csv(file_name, header=None).values)[:n].reshape((n,))
    
train_Y = numpy.asarray(pd.read_csv('data/y.csv', header=None).values)[:n]
train_X = numpy.transpose(train_X)

print(train_X.shape)
print(train_Y.shape)



threshold = 0.01

X = tf.placeholder(tf.float32,shape=(dimension))
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([dimension], 0, 1, seed=0))
b = tf.Variable(tf.zeros([1]))

y_pred = tf.add(tf.tensordot(X, W, axes=1), b)

cost = tf.pow(y_pred-Y, 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/gpu:0", "/gpu:1"])

ep_ = []
c_ = []
t_ = []


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    start = time.time()
    
    for epoch in range(maxEpochs):
        c = 0
        for i in range(n):
            c += sess.run(cost, feed_dict={X: train_X[i], Y:train_Y[i]})
        if epoch != 0 and abs(c - c_[-1]) < threshold:
            break
            
        ep_.append(epoch)
        c_.append(c)
        t_.append(time.time() - start)
        
        if epoch % verbose == 0:
            print("Epoch:", '%d' % (epoch), "cost=", c, "W =", sess.run(W), "b =", sess.run(b))
        for (x, y) in zip(train_X, train_Y):         
            sess.run(optimizer, feed_dict={X: x, Y: y})

    end = time.time()

    print("Training is done!")
    print("Time:", end - start)
    
print_out = [ep_, c_, t_]
print_out = list(map(list,zip(*print_out)))

file_name = str(n) + '.csv'
with open(file_name, 'w') as f:
    writer = csv.writer(f)
    for row in print_out:
        writer.writerow(row)