#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: example_rbm.py
#  DESCRIPTION: Example of use rbm.py 
#        USAGE: run it in python
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#         This file is part of DBN-Tensorflow.
#         DBN-Tensorflow is free software: you can redistribute it and/or modify
#         it under the terms of the GNU General Public License as published by
#         the Free Software Foundation, either version 3 of the License, or
#         (at your option) any later version.
#        
#         DBN-Tensorflow is distributed in the hope that it will be useful,
#         but WITHOUT ANY WARRANTY; without even the implied warranty of
#         MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#         GNU General Public License for more details.
#        
#         You should have received a copy of the GNU General Public License
#         along with DBN-Tensorflow.  If not, see <https://www.gnu.org/licenses/>.
#
#         BUGS:
#        NOTES:
#           Needs a lot of tests
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.21.
# =============================================================================


#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from mydataset import DBNDataset
from rbm import BinaryRBM
from tf_utils_dbn import TensorflowInitialize

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()



rbm = BinaryRBM(
     # Structure
     100,                # number of hidden units
     n_binary_units=28*28,              # number of visible binary units
     n_discriminative=10,            # number of labels (enables softmax units) 
     n_persistant_chains=100,         # enables Persistant CD (PCD) if > 0
     centering=True,
     centering_decay=0.98,
     # Learning
     minibatch_size=100,            # minibatch size
     contrastive_divergence_iter=1, # number of Gibbs sampling steps (for both CD and PCD)
     learning_rate=2e-3,            # initial learning rate
     learning_rate_decay=0.99,         # learning rate decay (is exponential for beter results) 
                                    # [theoretic 1/n is not that good]
     momentum=0.5,                   # float >= 0
     use_adam=True,                # use adam optimizer (default tf arguments)
     l1_regularization=0.0,           # l1 regularization (for better expalinability)
     l2_regularization=0.001,           # l2 regularization 

     sparsity_method='no',
     sparsity_target=0.5,
     sparsity_cost=1.,
     sparsity_variance=0.2,

     #target_loss='pseudo-likelihood-100',
     target_loss='pseudo-likelihood-1000',
     decrese_evaluation_limit=5,
     logged_losses_training=['reconstruction', 'mse', 'free-energy'],
     logged_losses_validation=['reconstruction', 'mse', 'free-energy', 'l2-regularization-mean', 'learning-rate'],

     # Generation of new samples - 0 means do not create this part of graph
     generative_gibbs_steps=10,
     name='RBM',
     verbose=2) # 0- no printed output 1: print results for each batch 2: print results for each minibatch


x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1]*x_val.shape[2]))
#x_val = x_val > (256/2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
#x_train = x_train > (256/2)

#import pdb; pdb.set_trace()
with TensorflowInitialize('example_rbm') as sess:
    def convert_probability(inp, lab):
        inp = tf.cast(inp, tf.float32)
        inp = tf.truediv(inp, 255.)
        inp = tf.distributions.Bernoulli(probs=inp, dtype=tf.float32).sample()
        return inp, lab
    def convert_binarize(inp, lab):
        inp = tf.cast(inp > 0, tf.float32)
        return inp, lab
    dataset = DBNDataset(nparray_validation=(x_val, y_val), nparray_train=(x_train, y_train), mapping=convert_binarize)
    '''
    dataset = DBNDataset(nparray_validation=(x_val, y_val), nparray_train=(x_train, y_train))
    '''


    rbm.run_pretraining(dataset, sess)
    rbm.plot_filters('tests/filters-10.png', (28,28))
    rbm.plot_generated_images('tests/images-10.png', (28,28), 100)
    rbm.plot_generated_images('tests/images-prob.png', (28,28), 100, prob=True)


