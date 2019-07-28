#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: rbm.py
#  DESCRIPTION:
#        USAGE:
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
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.03.
# =============================================================================
import numpy as np
import tensorflow as tf
import time

from tf_utils_dbn import BaseTensorFlowModel, SingletonMetaClass, TensorflowInitialize
from mydataset import DBNDataset

class ProbTensorflowCalculator(metaclass=SingletonMetaClass):
    """ Enables fast computations of probabilities on binary units (tensors) """
    def __init__(self, val):
        self.graph_created = {}

    def create_current_graph(self):
        with tf.name_scope('ProbabilityComputation'):
            inp = tf.placeholder(shape=[None, None], dtype=tf.float32, name='input')
            out = tf.reduce_sum(self.inp, axis=0) / tf.shape(self.inp)[0]
        return (inp, out)

    def __call__(self, nparray):
        try:
            (inp, out) = self.graph_created[tf.get_default_graph()]
        except KeyError:
            (inp, out) = self.create_current_graph()
            self.graph_created[tf.get_default_graph()] = (inp, out)
        return tf.get_default_session().run(out, {inp:nparray})

class EpochNumTraining(Exception):
    pass
class TimeoutTraining(Exception):
    pass


class BinaryRBM(BaseTensorFlowModel):
    """
    This class implements a Binary Restricted Boltzmann machine based on TensorFlow.

    minibatch_size: minibatch_size - <10,100>
    n_persistant_chains - normaly equal to minibatch_size
    contrastive_divergence_iter - for PCD small like 1, for CD depends if you want reliable reconstruction (10) or speed (1)

    learning_rate - for PCD very small around, for CD-1
    """
    TARGET_LOSSES = ['no', 'mse', 'reconstruction', 'free-energy', 'pseudo-likelihood'] # + pseudo-likelihood-N
    VALIDATION_LOSSES = [
            'learning-rate', 
            'l1-regularization-mean',
            'l2-regularization-mean',
            'mse', 'mse-split', 
            'reconstruction', 'reconstruction-split', 
            'free-energy', 
            'pseudo-likelihood'] # + pseudo-likelihood-N
    TRAIN_LOSSES = VALIDATION_LOSSES + ['derivative-sum']

    SPARSITY_METHODS=['no', 'normal', 'rate-distortion', 'quadratic']

    def __init__(self, 
             # Structure
             n_hidden_units,                # number of hidden units
             n_binary_units=0,              # number of visible binary units
             n_normalized_gaussian_units=0, # number of gaussian visible units (shoud be normalized to N(0,1))
             n_discriminative=0,            # number of labels (enables softmax units) 
             n_persistant_chains=0,         # enables Persistant CD (PCD) if > 0

             # Learning
             minibatch_size=100,            # minibatch size
             contrastive_divergence_iter=1, # number of Gibbs sampling steps (for both CD and PCD)
             centering=False,               # Centering 
             centering_decay=0.999,
             learning_rate=1e-3,            # initial learning rate
             learning_rate_decay=0,         # learning rate decay (is exponential for beter results) 
                                            # [theoretic 1/n is not that good]
             momentum=0.,                   # float >= 0
             use_adam=False,                # use adam optimizer (default tf arguments)
             l1_regularization=0,           # l1 regularization (for better expalinability)
             l2_regularization=0,           # l2 regularization 

             sparsity_method='no',          # sparsity method to use ['no', 'normal', TODO]
             sparsity_target=0.01,          # target probability
             sparsity_cost=0,               # sparsity multiplier
             sparsity_variance=0.9,         # for "normal" sparsity only

             # TODO: DBN:
             # TODO: top_down_pretraining
             # TODO: top_down_pretraining=False, [A fast learning algorithm for deep belief nets (Hinton, Osindero, Teh)]
                    

             # Learning stop conditions
             maximum_epochs=None,           # maximum epochs in training (separately pretraining and finetuning)
             maximum_time=None,             # maximum time (minutes) for training (separately pretraining and finetuning)
             target_loss='pseudo-likelihood-100',
             decrese_evaluation_limit=5,
                # For pretraining step - if target_loss is 'decrese_evaluation_limit' times worse on validation data
                # then there is backtracking to best parameters and pretraining step ends
                # avaiable losses are:
                #   'no':                   use maximum_epochs/time only (or infinite computation)
                #   'reconstruction':       use reconstruction error (cross-entroy)
                #   'reconstruction-split': use reconstruction error for each set of inputs separately (cannot be target_loss)
                #   'mse':                  use MSE
                #   'mse-split':            use MSE for each set of inputs separately (cannot be target_loss)
                #   'free-energy':          use free energy
                #   'pseudo-likelihood-N':  use (negative) pseudo-likelihood with random sampling of units (prefered method) N is number of samples
                #   'pseudo-likelihood':    use (negative) pseudo-likelihood (no sampling)
                #   'learning-rate'
             logged_losses_training=['reconstruction'],
             logged_losses_validation=['reconstruction'],

             # Generation of new samples - 0 means do not create this part of graph
             generative_gibbs_steps=0,

             name='RBM',
             verbose=0): # 0- no printed output 1: print results for each batch 2: print results for each minibatch
        super().__init__()
        
        # Learning 
        self.centering = centering
        assert isinstance(centering, bool)
        self.centering_decay = centering_decay
        assert isinstance(centering_decay, float) and 1. > centering_decay >= 0.
        self.contrastive_divergence_iter = contrastive_divergence_iter
        assert self.contrastive_divergence_iter > 0
        self.minibatch_size = minibatch_size
        assert self.minibatch_size > 0.
        self.initial_learning_rate = learning_rate
        assert self.initial_learning_rate > 0.
        self.l2_regularization = l2_regularization
        assert self.l2_regularization >= 0.
        self.l1_regularization = l1_regularization
        assert self.l1_regularization >= 0.
        self.momentum = momentum
        assert self.momentum >= 0.
        self.learning_rate_decay = float(learning_rate_decay)
        assert self.learning_rate_decay >= 0.
        self.use_adam = use_adam
        assert isinstance(self.use_adam, bool)
        assert self.use_adam or (not self.use_adam and self.momentum == 0.)

        # TODO: sparsity [_initialize_weights !!! if added]
        # SPARSITY
        self.sparsity_method = sparsity_method
        assert self.sparsity_method in self.SPARSITY_METHODS
        self.sparsity_target = sparsity_target
        assert 1. > self.sparsity_target > 0.
        self.sparsity_cost = sparsity_cost
        assert self.sparsity_cost > 0. or self.sparsity_method == 'no'  # use 'no' for disabling sparsity
        self.sparsity_variance= sparsity_variance
        assert 0. < self.sparsity_variance 


        

        # Structure
        self.n_binary_units = n_binary_units
        assert self.n_binary_units > 0
        self.n_hidden_units = n_hidden_units
        assert self.n_hidden_units > 0
        self.n_discriminative = n_discriminative 
        assert self.n_discriminative >= 0
        self.n_normalized_gaussian_units = n_normalized_gaussian_units
        assert self.n_normalized_gaussian_units >= 0
        self.n_persistant_chains = n_persistant_chains if n_persistant_chains is not None else 0
        assert self.n_persistant_chains >= 0



        # Learning stop conditions
        self.maximum_epochs = maximum_epochs if maximum_epochs is not None and maximum_epochs >= 0 else None
        assert self.maximum_epochs is None or self.maximum_epochs >= 0
        self.maximum_time = maximum_time if maximum_time is not None and maximum_time >= 0 else None
        assert self.maximum_time is None or self.maximum_time >= 0
        self.logged_losses_training = logged_losses_training 
        assert isinstance(self.logged_losses_training, list)
        assert all((x in self.TRAIN_LOSSES or x.startswith('pseudo-likelihood-') for x in self.logged_losses_training))
        self.logged_losses_validation = logged_losses_validation 
        assert isinstance(self.logged_losses_validation, list)
        assert all((x in self.VALIDATION_LOSSES or x.startswith('pseudo-likelihood-') for x in self.logged_losses_validation))
        self.decrese_evaluation_limit = decrese_evaluation_limit
        assert self.decrese_evaluation_limit >= 0
        self.target_loss = target_loss
        assert self.target_loss in self.TARGET_LOSSES or self.target_loss.startswith('pseudo-likelihood-')

        # Generative properties
        self.generative_gibbs_steps = generative_gibbs_steps
        assert self.generative_gibbs_steps >= 0

        # Other
        self.verbose = 1 if isinstance(verbose, bool) and verbose else verbose
        assert isinstance(self.verbose, int)
        self.name = name
        assert isinstance(self.name, str)

        # States
        self.build_complete = False
        self.variables_to_initialize = []
        # load / save 
        self.checkpoint_operations = []
        self.checkpoint_values = []
        self.checkpoint_placeholders = []
        # summaries
        self.validation_summaries = []
        self.training_summaries = []

    @property
    def is_PERSISTENT(self):
        return self.n_persistant_chains > 0
    @property
    def is_DISCRIMINATIVE(self):
        return self.n_discriminative > 0
    @property
    def is_GAUSSIAN(self):
        return self.n_normalized_gaussian_units > 0
    @property
    def is_BINARY(self):
        return self.n_binary_units > 0
    @property
    def is_CENTERED(self):
        return self.centering
    @property
    def n_visible_units(self):
        return self.n_discriminative + self.n_normalized_gaussian_units + self.n_binary_units






    def _initialize_weights(self, weights=None, visible_sample=None, label_sample=None):
        """ Load or initialize weights """
        assert label_sample is None or self.is_DISCRIMINATIVE

        if weights:
            # TODO renew
            # LOAD weights
            with name_scope('LoadWeights'):
                for attr_name, value in weights.items():
                    if value is not None:
                        self.__setattr__(attr_name, tf.Variable(value))
                    else:
                        self.__setattr__(attr_name, None)
        else:
            # Create weights ...
            # (Hinton 2010) initialization recomendations
            # (I don't think xavier initialization helps RBM)
            with tf.name_scope('InitWeights'):
                if visible_sample is None:
                    generator_b_vis_sample = tf.zeros(shape=[self.n_binary_units + self.n_normalized_gaussian_units], 
                            name='bVisInitConst')
                else:
                    # better to initialize with log(pi/1(1-pi))  (Hinton 2010)
                    visible_probs = tf.constant(ProbTensorflowCalculator()(visible_sample), dtype=tf.float32, 
                            shape=[self.n_normalized_gaussian_units + self.n_binary_units])
                    # 1/0 -> 0.
                    a = tf.where(tf.equal(visible_probs, 1.), 
                            tf.zeros_like(visible_probs), visible_probs/(1-visible_probs))
                    # log(0) -> 0.
                    generator_b_vis_sample = tf.where(tf.equal(a, 0.), tf.zeros_like(a), tf.log(a), name='bVisInitConst')

                if label_sample is None:
                    generator_b_lab_sample = tf.zeros(shape=[self.n_discriminative], name='bLabInitConst')
                else:
                    # better to initialize with log(pi/1(1-pi))  (Hinton 2010)
                    label_probs = tf.constant(ProbTensorflowCalculator()(label_sample), 
                            dtype=tf.float32, shape=[self.n_discriminative])
                    # 1/0 -> 0.
                    a = tf.where(tf.equal(label_probs, 1.), 
                            tf.zeros_like(label_probs), label_probs/(1-label_probs))
                    # log(0) -> 0.
                    generator_b_lab_sample = tf.where(tf.equal(a, 0.), tf.zeros_like(a), tf.log(a), name='bLabInitConst')

                generator_b_sample = tf.concat([generator_b_lab_sample, generator_b_vis_sample], axis=0, name='bInitConst')
                generator_W_sample = tf.distributions.Normal(loc=0., scale=0.01).sample(
                        sample_shape=[self.n_hidden_units, self.n_visible_units], name='wInitConst')
                # If I want to use sparsity this shoud be changed to log(t/(1-t) !!!
                generator_c_sample = tf.zeros(shape=[self.n_hidden_units], name='cInitConst')

                self.persistant_chains = None

                if self.is_CENTERED:
                    # TODO change to first data 
                    generator_b_centered = tf.zeros([self.n_visible_units], dtype=tf.float32)
                    generator_c_centered = tf.ones([self.n_hidden_units], dtype=tf.float32) * 0.5


            self.W = tf.Variable(generator_W_sample, 
                    expected_shape=[self.n_hidden_units, self.n_visible_units], 
                    dtype=tf.float32, name='W')
            self.b = tf.Variable(generator_b_sample, 
                    expected_shape=[self.n_visible_units], dtype=tf.float32,
                    name='b')
            self.c = tf.Variable(generator_c_sample, 
                    expected_shape=[self.n_hidden_units], dtype=tf.float32,
                    name='c')
            if self.is_CENTERED:
                self.b_center = tf.Variable(generator_b_centered,
                    expected_shape=[self.n_visible_units], dtype=tf.float32,
                    name='b_center')
                self.c_center = tf.Variable(generator_c_centered, 
                    expected_shape=[self.n_hidden_units], dtype=tf.float32,
                    name='c_center')

        if self.is_PERSISTENT:
            with tf.name_scope('InitWeights'):
                l = []
                if self.is_DISCRIMINATIVE:
                    a = tf.distributions.Categorical(probs=[1./self.n_discriminative]*self.n_discriminative).sample(
                            sample_shape=[self.n_persistant_chains])
                    a = tf.one_hot(a, self.n_discriminative)
                    l.append(a)
                if self.is_BINARY:
                    b = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32).sample(
                            sample_shape=[self.n_persistant_chains, self.n_binary_units])
                    l.append(b)
                if self.is_GAUSSIAN:
                    c = tf.distributions.Normal(loc=0., scale=1.).sample(
                            sample_shape=[self.n_persistant_chains, self.n_normalized_gaussian_units])
                    l.append(c)
                generator_persistant_chains = tf.concat(l, axis=1)

            self.persistant_chains = tf.Variable(generator_persistant_chains, name='persistantChains')
            self._add_to_init(self.persistant_chains)

        if self.is_CENTERED:
            self._add_to_init(self.b_center, self.c_center)
            self._add_to_checkpoint(self.b_center, self.c_center)

        # load adn save state of network
        self._add_to_checkpoint(self.W, self.b, self.c)
        # add initializers
        self._add_to_init(self.W, self.b, self.c)

        # shortcuts for easy access to different parts of network
        self.W_labels = self.W[:,:self.n_discriminative]
        self.W_features = self.W[:,self.n_discriminative:]
        self.b_labels = self.b[:self.n_discriminative]
        self.b_features = self.b[self.n_discriminative:]



    def _add_to_checkpoint(self, *variables):
        """ add variables to checkpoint (load/save) and create operation to load it"""
        with tf.name_scope('CheckpointLoad'):
            for variable in variables:
                ph = tf.placeholder(variable.dtype, variable.shape)
                op = tf.assign(variable, ph)
                self.checkpoint_values.append(variable)
                self.checkpoint_placeholders.append(ph)
                self.checkpoint_operations.append(op)

    def _add_to_init(self, *operations):
        """ adds this operations to init """
        for operation in operations:
            assert hasattr(operation, 'initializer')
            self.variables_to_initialize.append(operation)

    def _save_end_of_chain(self, last_visible_sample):
        return tf.assign(self.persistant_chains, last_visible_sample)
    def _split_visible(self, visible):
        """ split softmax, binary and gaussian units """
        discrim = visible[:self.n_discriminative]
        visible = visible[self.n_discriminative:]
        binary = visible[:self.n_binary_units]
        visible = visible[self.n_binary_units:]
        gaussian = visible
        return (discrim, binary, gaussian)


    def _sample_visible_units(self, hidden_sample):
        """ tf sampling of visible units """
        with tf.name_scope('VisibleUnitsSampling'):
            if self.is_CENTERED:
                visible_preactivation = tf.matmul(hidden_sample - self.c_center, self.W) + self.b
            else:
                visible_preactivation = tf.matmul(hidden_sample, self.W) + self.b
            original_visible_preactivation = visible_preactivation
            probs, samples = [], []

            if self.is_DISCRIMINATIVE:
                with tf.name_scope('label'):
                    # 1. split
                    label_preactivation = visible_preactivation[:, :self.n_discriminative]
                    # 2. Learning for softmax unit is the same as for binary (Hinton 2010)
                    label_probs = tf.nn.softmax(label_preactivation)
                    # 3. sampling
                    label_sample = tf.distributions.Bernoulli(probs=label_probs, dtype=tf.float32).sample()
                visible_preactivation = visible_preactivation[:, self.n_discriminative:]
                probs.append(label_probs)
                samples.append(label_sample)

            if self.is_BINARY:
                with tf.name_scope('binary'):
                    binary_preactivation = visible_preactivation[:, :self.n_binary_units]
                    binary_probs = tf.sigmoid(binary_preactivation)
                    binary_sample = tf.distributions.Bernoulli(probs=binary_probs, dtype=tf.float32).sample()
                visible_preactivation = visible_preactivation[:, self.n_binary_units:]
                probs.append(binary_probs)
                samples.append(binary_sample)

            if self.is_GAUSSIAN:
                with tf.name_scope('gauss'):
                    # 1. split
                    gauss_preactivation = visible_preactivation[:, :self.n_normalized_gaussian_units]
                    # 2. Learning for softmax unit is the same as for binary (Hinton 2010)
                    #gauss_probs = None
                    gauss_sample = gauss_preactivation
                visible_preactivation = visible_preactivation[:, self.n_normalized_gaussian_units:]
                #probs.append(gauss_probs)
                samples.append(gauss_sample)
            assert visible_preactivation.shape[1] == 0


            visible_probs = tf.concat(probs, axis=1)
            visible_sample = tf.concat(samples, axis=1)
            return (original_visible_preactivation, visible_probs, visible_sample)

    def _sample_hidden_units(self, visible_units):
        """ tf sampling of hidden units (all binary) """
        with tf.name_scope('HiddenUnitsSampling'):
            if self.is_CENTERED:
                hidden_preactivation = tf.matmul(visible_units - self.b_center, self.W, transpose_b=True) + self.c
            else:
                hidden_preactivation = tf.matmul(visible_units, self.W, transpose_b=True) + self.c
            hidden_activation = tf.sigmoid(hidden_preactivation)
            hidden_sample = tf.distributions.Bernoulli(probs=hidden_activation, dtype=tf.float32).sample()
            return (hidden_preactivation, hidden_activation, hidden_sample)

    def _nan_check(self, a):
        # TODO remove
        return tf.Print(a, [tf.where(tf.is_nan(a))])

    def _compute_gradient(self, 
            visible_units, first_hidden_activation, 
            visible_sample, hidden_activation):
        """ compute gradient with/withou centering + adds regularizations (both l and sparsity) """
        with tf.name_scope('GradientComputation'):
            if self.is_CENTERED:
                negative_gradient = tf.reduce_mean(tf.matmul(
                    tf.expand_dims(hidden_activation - tf.expand_dims(self.c_center, 0), 2),  # [N, U, 1]
                    tf.expand_dims(visible_sample - tf.expand_dims(self.b_center, 0), 1))     # [N, 1, V]
                    , axis=0)
                positive_gradient = tf.reduce_mean(tf.matmul(
                    tf.expand_dims(first_hidden_activation - tf.expand_dims(self.c_center, 0), 2),  # [N, U, 1]
                    tf.expand_dims(visible_units - tf.expand_dims(self.b_center, 0), 1))  # [N, 1, V]
                    , axis=0)
                compute_delta_W = positive_gradient - negative_gradient
                compute_delta_b = tf.reduce_mean(visible_units, 0) - tf.reduce_mean(visible_sample, 0)
                compute_delta_c = tf.reduce_mean(first_hidden_activation, 0) - tf.reduce_mean(hidden_activation, 0)

                compute_delta_b -= tf.squeeze(tf.matmul(compute_delta_W, tf.expand_dims(self.c_center,1), transpose_a=True), axis=1)
                compute_delta_c -= tf.squeeze(tf.matmul(compute_delta_W, tf.expand_dims(self.b_center,1)), axis=1)
            else:
                compute_delta_b = tf.reduce_mean(visible_units, 0) - tf.reduce_mean(visible_sample, 0)
                compute_delta_c = tf.reduce_mean(first_hidden_activation, 0) - tf.reduce_mean(hidden_activation, 0)
                negative_gradient = tf.reduce_mean(
                    tf.matmul(tf.expand_dims(hidden_activation, 2),  # [N, U, 1]
                              tf.expand_dims(visible_sample, 1))     # [N, 1, V]
                    , axis=0)
                positive_gradient = tf.reduce_mean(
                    tf.matmul(tf.expand_dims(first_hidden_activation, 2),  # [N, U, 1]
                              tf.expand_dims(visible_units, 1))  # [N, 1, V]
                    , axis=0)
                compute_delta_W = positive_gradient - negative_gradient

        with tf.name_scope('Regularization'):
            (a,b,c) = self._compute_regularization(visible_units, first_hidden_activation)
            compute_delta_W += a
            compute_delta_b += b
            compute_delta_c += c
        return compute_delta_W, compute_delta_b, compute_delta_c

    def _choose_optimizer(self, grads):
        """ create self.training for pretraining network """
        if self.learning_rate_decay > 0.:
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, 
                self.global_step, 1, self.learning_rate_decay)
        else:
            self.learning_rate = tf.constant(self.initial_learning_rate, dtype=tf.float32)

        # choose optimizer
        if self.use_adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        else:
            if self.momentum > 0:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # give gradients to optimizers
        self.training = optimizer.apply_gradients(grads, global_step=self.global_step)

        # init optimizers
        self._add_to_init(*[optimizer.get_slot(var, slotname) for slotname in optimizer.get_slot_names() 
            for var in [self.W, self.b, self.c]])
        if self.use_adam:
            self._add_to_init(*list(optimizer._get_beta_accumulators()))


    def _create_update_for_centering(self, visible, hidden):
        """ adds to training also centering update ops """
        visible = tf.reduce_mean(visible, axis=0)
        hidden = tf.reduce_mean(hidden, axis=0)
        with tf.control_dependencies([self.training]):
            op_change_b = tf.assign_sub(self.b_center, (1-self.centering_decay)*(self.b_center - visible))
            op_change_c = tf.assign_sub(self.c_center, (1-self.centering_decay)*(self.c_center - hidden))

        self.training = tf.group(self.training, op_change_b, op_change_c)

    

    def _build_model(self, data_iterator, weights=None):
        """ Builds TensorFlow model.  """
        assert not self.build_complete
        self._initialize_weights(weights)

        if self.is_DISCRIMINATIVE:
            self.visible_units_placeholder, self.label_placeholder = data_iterator.get_next()

            with tf.name_scope('createVisible'):
                assert(self.label_placeholder.dtype == tf.int32)
                labels = tf.one_hot(self.label_placeholder, self.n_discriminative, dtype=tf.float32)
                visible_units = tf.concat([labels, self.visible_units_placeholder], axis=1)
        else:
            self.visible_units_placeholder = data_iterator.get_next()
            visible_units = self.visible_units_placeholder

        # sample Hidden units
        (first_hidden_preactivation, first_hidden_activation, first_hidden_sample) = \
            self._sample_hidden_units(visible_units)
        # sample Visible units
        (first_visible_preactivation, first_visible_activation, first_visible_sample) = \
            self._sample_visible_units(first_hidden_sample)

        # Negative gradient - (PCD starts differently)
        with tf.name_scope('NegativeGradient'):
            if self.is_PERSISTENT:
                visible_sample = self.persistant_chains
                (hidden_preactivation, hidden_activation, hidden_sample) = \
                    self._sample_hidden_units(visible_sample)
            else:
                visible_preactivation, visible_activation, visible_sample = (first_visible_preactivation, first_visible_activation, first_visible_sample)

            with tf.name_scope('GibbsSampling'):
                for t in range(self.contrastive_divergence_iter):
                    if self.is_PERSISTENT or t > 0:
                        (visible_preactivation, visible_activation, visible_sample) = \
                            self._sample_visible_units(hidden_sample)
                    (hidden_preactivation, hidden_activation, hidden_sample) = \
                        self._sample_hidden_units(visible_sample)

        with tf.name_scope('TrainUpdates'):
            self.global_step = tf.train.get_or_create_global_step()
            self._add_to_init(self.global_step)

            compute_delta_W, compute_delta_b, compute_delta_c = self._compute_gradient(
                    visible_units, first_hidden_activation,
                    visible_sample, hidden_activation)
            self.sum_of_derivations = sum(tf.reduce_sum(tf.abs(x)) for x in [compute_delta_b, compute_delta_c, compute_delta_W])

            self._choose_optimizer([(-compute_delta_W, self.W), (-compute_delta_b, self.b), (-compute_delta_c, self.c) ])

            if self.is_PERSISTENT:
                with tf.control_dependencies([self.training]):
                    self.training = tf.group(self.training, self._save_end_of_chain(visible_sample))
            if self.is_CENTERED:
                self._create_update_for_centering(visible_units, first_hidden_activation)

        self._generative_parts_of_rbm()
        self._generative_parts_of_rbm_one_class_only()
        self._build_cost_methods(visible_units, first_hidden_preactivation, first_visible_preactivation, first_visible_sample)

        tf.get_default_session().run(tf.variables_initializer(self.variables_to_initialize))
        self.variables_to_initialize = []
        self.build_complete = True

    def _compute_regularization(self, visible, first_hidden_activation):
        compute_delta_W, compute_delta_b, compute_delta_c = tf.zeros_like(self.W), tf.zeros_like(self.b), tf.zeros_like(self.c)
        # L2
        if self.l2_regularization > 0:
            # bias is not regulated because it does not contribute to overfitting much
            with tf.name_scope('l2'):
                a = self.l2_regularization*self.W
                compute_delta_W += -a
                self.l2_regularization_mean = tf.reduce_mean(a)
        # L1
        if self.l1_regularization > 0:
            with tf.name_scope('l1'):
                a = self.l1_regularization*tf.sign(self.W)
                compute_delta_W += -a
                self.l1_regularization_mean = tf.reduce_mean(a)

        # Sparsity methods
        if self.sparsity_method == 'no':
            pass
        elif self.sparsity_method == 'rate-distortion':
            g = tf.multiply(first_hidden_activation, (1-first_hidden_activation))
            m = tf.matmul(g, visible, transpose_a=True) * tf.realdiv(1.,tf.cast(tf.shape(visible)[0], tf.float32))

            compute_delta_W += -self.sparsity_cost*m
            compute_delta_c += -self.sparsity_cost*tf.reduce_mean(g, axis=0)
        elif self.sparsity_method == 'normal':
            a = tf.reduce_mean(self.sparsity_target - first_hidden_activation, axis=0)
            aa = tf.tile(tf.expand_dims(a, axis=1), [1,self.n_visible_units])

            b = tf.multiply(first_hidden_activation, (1-first_hidden_activation))
            bb = tf.matmul(b, visible, transpose_a=True) * tf.realdiv(1.,tf.cast(tf.shape(visible)[0], tf.float32))
            
            c = tf.reduce_mean(b, axis=0)

            d = tf.distributions.Normal(loc=self.sparsity_target, scale=self.sparsity_variance).prob(
                tf.reduce_mean(first_hidden_activation, axis=0))
            dd = tf.tile(tf.expand_dims(d, axis=1), [1,self.n_visible_units])

            compute_delta_W += self.sparsity_cost*tf.multiply(aa, tf.multiply(bb, dd))
            compute_delta_c += self.sparsity_cost*tf.multiply(a, tf.multiply(c, d))
        elif self.sparsity_method == 'quadratic':
            a = tf.reduce_mean(self.sparsity_target - first_hidden_activation, axis=0)
            aa = tf.tile(tf.expand_dims(a, axis=1), [1,self.n_visible_units])

            b = tf.multiply(first_hidden_activation, (1-first_hidden_activation))
            bb = tf.matmul(b, visible, transpose_a=True) * tf.realdiv(1.,tf.cast(tf.shape(visible)[0], tf.float32))

            c = tf.reduce_mean(b, axis=0)

            compute_delta_W += self.sparsity_cost*tf.multiply(aa, bb)
            compute_delta_c += self.sparsity_cost*tf.multiply(a,c)
        else:
            raise NotImplementedError('Sparsity named "{}" is not implemented'.format(self.sparsity_method))

        return (compute_delta_W, compute_delta_b, compute_delta_c)



    def _build_cost_methods(self, visible_units, first_hidden_preactivation, first_visible_preactivation, first_visible_sample):
        """ create all posible summaries """
        with tf.name_scope('Metrics'):
            with tf.contrib.summary.record_summaries_every_n_global_steps(1):
                with tf.name_scope('ValidationLosses'):
                    for loss_name in self.logged_losses_validation + [self.target_loss]:
                        losses, labels = self._choose_cost_by_name(loss_name, visible_units, first_hidden_preactivation, first_visible_preactivation, first_visible_sample)
                        for loss, name in zip(losses, labels):
                            if loss is None:
                                continue
                            summary = TensorflowInitialize().add_scalar_summary('validation-'+name, loss)
                            self.validation_summaries.append(summary)
                            if loss_name == self.target_loss:
                                self.evaluation = loss
                    self.validation_summaries = [tf.summary.merge(self.validation_summaries)]

                with tf.name_scope('TrainLosses'):
                    for loss_name in self.logged_losses_training:
                        losses, labels = self._choose_cost_by_name(loss_name, visible_units, first_hidden_preactivation, first_visible_preactivation, first_visible_sample)
                        for loss, name in zip(losses, labels):
                            if loss is None:
                                continue
                            summary = TensorflowInitialize().add_scalar_summary('training-'+name, loss)
                            self.training_summaries.append(summary)
                    self.training_summaries = [tf.summary.merge(self.training_summaries)]

    def _choose_cost_by_name(self, loss_name, visible_units, first_hidden_preactivation, first_visible_preactivation, first_visible_sample):
        if loss_name == 'free-energy':
            loss = self._create_free_energy_cost(visible_units, hidden_preactivation=first_hidden_preactivation)
            labels = ['free_energy']
        elif loss_name == 'reconstruction':
            loss = self._create_reconstruction_cost(visible_units, first_visible_preactivation)
            labels = ['cross-entropy']
        elif loss_name == 'reconstruction-split':
            (ll, bl, gl) = self._create_reconstruction_cost(visible_units, first_visible_preactivation, extra=True)
            loss = [ll,bl,gl]
            labels = ['label-cross-entropy', 'binary-cross-entropy', 'gaussian-cross-entropy']
        elif loss_name == 'mse':
            loss = self._create_mse_cost(visible_units, first_visible_sample)
            labels = ['mse']
        elif loss_name == 'mse-split':
            (ll, bl, gl) = self._create_mse_cost(visible_units, first_visible_sample, extra=True)
            loss = [ll,bl,gl]
            labels = ['label-mse', 'binary-mse', 'gaussian-mse']
        elif loss_name == 'pseudo-likelihood':
            loss = self._create_pseudo_likelihood_cost(visible_units)
            labels = ['full-neg-pseudo-likelihood']
        elif loss_name == 'learning-rate':
            loss = self.learning_rate
            labels = ['learning-rate']
        elif loss_name == 'derivative-sum':
            loss = self.sum_of_derivations
            labels = ['derivative-sum']
        elif loss_name == 'l1-regularization-mean':
            loss = self.l1_regularization_mean
            labels = ['l1-regularization-mean']
        elif loss_name == 'l2-regularization-mean':
            loss = self.l2_regularization_mean
            labels = ['l2-regularization-mean']
        elif loss_name.startswith('pseudo-likelihood-'):
            Nsamples = int(loss_name[len('pseudo_likelihood-'):])
            loss = self._create_pseudo_likelihood_cost(visible_units, samples=Nsamples)
            labels=['{}-prob-neg-pseudo-likelihood'.format(Nsamples)]

        else:
            raise NotImplementedError('Loss "' + loss_name + '" is not implemented')
        return ([loss] if not isinstance(loss, list) else loss), labels


    def _convert_tensor_to_bool(self, tensor):
        return tensor > 0.5


    def _create_free_energy_cost(self, visible, hidden_preactivation=None, per_sample=False):
        """ Computes the RBM free energy.  [Learning Deep Architectures for AI by Yoshua Bengio] """
        # TODO Centering - different computation
        with tf.name_scope('FreeEnergy'):
            if self.is_CENTERED:
                if hidden_preactivation is None:
                    hidden_preactivation = tf.matmul(visible-tf.expand_dims(self.b_center, 0), self.W, transpose_b=True) + self.c

                result = -tf.matmul(visible - tf.expand_dims(self.b_center,0), tf.expand_dims(self.b,1)) # [N,1]
                result = tf.squeeze(result, axis=1)

                m = 1+tf.exp(hidden_preactivation)
                c = -tf.reduce_sum(tf.log(m), axis=1)
                result = result + c
            else:
                if hidden_preactivation is None:
                    hidden_preactivation = tf.matmul(visible, self.W, transpose_b=True) + self.c

                result = -tf.matmul(visible, tf.expand_dims(self.b,1)) # [N,1]
                result = tf.squeeze(result, axis=1)

                m = 1+tf.exp(hidden_preactivation)
                c = -tf.reduce_sum(tf.log(m), axis=1)
                result = result + c

            if per_sample:
                return result
            else:
                return tf.reduce_mean(result)

    def _create_pseudo_likelihood_cost(self, visible, samples=None):
        """ Computes pseudo likelihood error for BINARY-DISCRIMINATIVE-RBM (not gaussian) with or without aproximations """
        assert tf.VERSION[0] == '1'

        with tf.name_scope('PseudoLikelihood'):
            N = tf.shape(visible)[0]

            if samples is None:
                with tf.name_scope('createPoints'):
                    change_units = tf.one_hot(tf.range(self.n_binary_units+ self.n_discriminative), 
                            self.n_discriminative + self.n_binary_units, dtype=tf.bool, on_value=True, off_value=False)
                    change_units = tf.tile(tf.expand_dims(change_units, 2), [1, 1, N])
                    change_units = tf.transpose(tf.reshape(change_units, [self.n_binary_units + self.n_discriminative, -1]))

                    Ntile = self.n_binary_units + self.n_discriminative
                    Nextra = None
            else:
                with tf.name_scope('samplePoints'):
                    sampleN = tf.constant(samples, dtype=tf.int32, shape=[], name="SampleNum")

                    change_units = tf.random_uniform(shape=[sampleN], 
                            maxval=self.n_binary_units+self.n_discriminative-1, minval=0, dtype=tf.int32)
                    change_units = tf.one_hot(change_units, self.n_binary_units + self.n_discriminative, axis=1,
                            on_value=True, off_value=False, dtype=tf.bool)

                    Ntile = tf.floordiv(sampleN, N)
                    Nextra = tf.floormod(sampleN, N)
            # change_units is [None, self.n_binary_units + self.n_discriminative] tensor of Boolean values


            with tf.name_scope('ExpandVisible'):
                visible_bool = self._convert_tensor_to_bool(visible[:,:self.n_discriminative+self.n_binary_units])
                visible_gauss = visible[:,self.n_discriminative+self.n_binary_units:]

                new_visible_units_bool = tf.tile(visible_bool, [Ntile, 1])
                new_visible_units_gauss = tf.tile(visible_gauss, [Ntile, 1])

                if Nextra is not None:
                    new_visible_units_bool = tf.concat(
                            [new_visible_units_bool, tf.gather(visible_bool, tf.range(Nextra, dtype=tf.int32))], axis=0)
                    new_visible_units_gauss = tf.concat(
                            [new_visible_units_gauss, tf.gather(visible_gauss, tf.range(Nextra, dtype=tf.int32))], axis=0)

            new_visible_units_bool = tf.cast(tf.logical_xor(new_visible_units_bool, change_units), tf.float32)
            new_visible_units = tf.concat([new_visible_units_bool, new_visible_units_gauss], axis=1)
            new_visible_units_free_energy = self._create_free_energy_cost(new_visible_units, per_sample=True)


            with tf.name_scope('ExpandFE'):
                visible_units_free_energy = self._create_free_energy_cost(visible, per_sample=True) 
                expanded_visible_units_free_energy = tf.tile(visible_units_free_energy, [Ntile])
                if Nextra is not None:
                    expanded_visible_units_free_energy = tf.concat(
                        [expanded_visible_units_free_energy, tf.gather(visible_units_free_energy, 
                            tf.range(Nextra, dtype=tf.int32))], axis=0)

            # e^-FE/(e^-FE + e^-~FE) = 1/(1 + e^-(~FE-FE)) and this is sigmoid basicaly
            # partition function will be cancelled in division
            # need to increse value by number of visible layers because it only samples one change
            r = self.n_visible_units*tf.reduce_mean(
                    tf.log_sigmoid(new_visible_units_free_energy-expanded_visible_units_free_energy), axis=0)
            # minimalization ... (so this is negative logpseudolikelihood)
            return -r

    def _create_reconstruction_cost(self, labels, logits, extra=False):
        """ Computes reconstruction error for RBM (without gaussian units) """
        with tf.name_scope('ReconstructionError'):
            llab, lbin, lgauss = self._split_visible(labels)
            rlab, rbin, rgauss = self._split_visible(logits)
            complete_result = None

            if self.is_DISCRIMINATIVE:
                label_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=llab, logits=rlab)
                label_cross_entropy = tf.reduce_mean(label_cross_entropy, axis=0)
                complete_result = label_cross_entropy if complete_result is None else complete_result + label_cross_entropy
            else:
                label_cross_entropy = None


            if self.is_BINARY:
                binary_cross_entropy= tf.nn.sigmoid_cross_entropy_with_logits(labels=lbin, logits=rbin)
                binary_cross_entropy = tf.reduce_mean(tf.reduce_sum(binary_cross_entropy, axis=1), axis=0)
                complete_result = binary_cross_entropy if complete_result is None else complete_result + binary_cross_entropy
            else:
                binary_cross_entropy = None

            if self.is_GAUSSIAN and extra:
                # TODO this is only test
                # labels shoud be normalized to have 1. std, logits can be different
                gaussian_cross_entropy = tf.distributions.Normal(loc=lgauss, scale=1.).cross_entropy(
                    tf.distributions.Normal(loc=rgauss, scale=1.))
                gaussian_cross_entropy = tf.reduce_mean(tf.reduce_sum(gaussian_cross_entropy, axis=1), axis=0)
            else:
                gaussian_cross_entropy = None

            if extra:
                return (label_cross_entropy, binary_cross_entropy, gaussian_cross_entropy)
            else:
                return complete_result

    def _create_mse_cost(self, labels, reconstruction, extra=False):
        """ MSE of two visible batches, extra - split to binary gaussian and label parts"""
        with tf.name_scope('MSE'):
            if extra:
                llab, lbin, lgauss = self._split_visible(labels)
                rlab, rbin, rgauss = self._split_visible(reconstruction)

                loss_gaussian = tf.losses.mean_squared_error(lgauss, rgauss)
                loss_binary = tf.losses.mean_squared_error(lbin, rbin)
                loss_label = tf.losses.mean_squared_error(llab, rlab)
                return (loss_label, loss_binary, loss_gaussian)
            else:
                return tf.losses.mean_squared_error(labels, reconstruction)

    def _generative_parts_of_rbm(self):
        if self.generative_gibbs_steps == 0:
            return

        with tf.name_scope('Generative'):
            # number of samples to generate
            self.n_samples = tf.placeholder(tf.int32, [], 'numberOfSamples')

            first_hidden_sample_generator = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
            first_hidden_sample_generator = first_hidden_sample_generator.sample(
                    sample_shape=[self.n_samples, self.n_hidden_units])

            with tf.name_scope('Gibbs'):
                hidden_sample = first_hidden_sample_generator
                for i in range(self.generative_gibbs_steps-1):
                    _, visible_sample_prob, visible_sample = self._sample_visible_units(hidden_sample)
                    _, _, hidden_sample = self._sample_hidden_units(visible_sample)

            self.generated_visible_layer_prob = visible_sample_prob
            self.generated_visible_layer = visible_sample

    def _generative_parts_of_rbm_one_class_only(self):
        """ this shoud generate one class only """
        if self.generative_gibbs_steps == 0 or not self.is_DISCRIMINATIVE:
            return

        with tf.name_scope('Generative'):
            #self.n_samples 
            self.generate_class = tf.placeholder(tf.int32, [], 'classToGenerate')

            label_part = tf.one_hot(self.generate_class, self.n_discriminative, dtype=tf.float32)
            label_part = tf.tile(tf.expand_dims(label_part, axis=0), [self.n_samples, 1])

            class_zeroes_vector = tf.concat(
                [label_part, tf.zeros((self.n_samples, self.n_binary_units+self.n_normalized_gaussian_units), dtype=tf.float32)], 
                axis=1)

            _, _, first_hidden_sample_generator_for_class = self._sample_hidden_units(class_zeroes_vector)

            with tf.name_scope('Gibbs'):
                hidden_sample = first_hidden_sample_generator_for_class
                for i in range(self.generative_gibbs_steps-1):
                    _, visible_sample_prob, visible_sample = self._sample_visible_units(hidden_sample)
                    visible_sample = tf.concat([label_part, visible_sample[:, self.n_discriminative:]], axis=1)

                    _, _, hidden_sample = self._sample_hidden_units(visible_sample)

            self.generated_visible_layer_class_prob = visible_sample_prob
            self.generated_visible_layer_class = visible_sample


    def generate_visible(self, n, specific_class=None, prob=False):
        if specific_class is None:
            if prob:
                a = tf.get_default_session().run(self.generated_visible_layer_prob, feed_dict={self.n_samples:n})
            else:
                a = tf.get_default_session().run(self.generated_visible_layer, feed_dict={self.n_samples:n})
        else:
            if prob:
                a = tf.get_default_session().run(self.generated_visible_layer_class_prob, feed_dict={self.n_samples:n, self.generate_class:specific_class})
            else:
                a = tf.get_default_session().run(self.generated_visible_layer_class, feed_dict={self.n_samples:n, self.generate_class:specific_class})
        return a

    def generated_visible_layer_from_hidden(self):
        """ this shoud be part of graph """
        raise NotImplementedError


    





    def run_pretraining(self, dbn_dataset, summary=None):
        """ Pretrain a model given data iterator.  """
        self._build_model(dbn_dataset.get_iterator())
        self._pretrain_model(dbn_dataset, summary)


    def _pretrain_model(self, dbn_dataset, summary_obj=None):
        """ Pretrains model given dataset """
        initial_time = time.time() # time since epoch (sec)
        epoch_number = 0

        evaluation_best = None
        evaluation_coutdown = 0
        evaluation_best_checkpoint = None

        if summary_obj is not None:
            summary_obj.init_summaries()

        try:
            while True:
                # set up validation data
                dbn_dataset.set_up_validation_data()
                # run validation
                evaluation_now, global_step, *state_now = tf.get_default_session().run(
                        [self.evaluation, self.global_step] + self.checkpoint_values + self.validation_summaries)
                # remove summaries from state
                state_now, summaries_now = state_now[:len(self.checkpoint_placeholders)], state_now[len(self.checkpoint_placeholders):]

                if summary_obj is not None:
                    summary_obj.pass_summaries(global_step, *summaries_now)

                if evaluation_best is None or evaluation_now < evaluation_best:
                    if self.verbose >= 2 and evaluation_best is not None:
                        print("{}:\t\tEpoch {} finished with {} validation error (better {} times)".format(
                            self.name, epoch_number, evaluation_now, evaluation_best/evaluation_now))
                    elif self.verbose >= 2:
                        print("{}:\t\tEpoch {} finished with {} validation error".format(self.name, epoch_number, evaluation_now))

                    evaluation_best = evaluation_now
                    evaluation_coutdown = 0
                    evaluation_best_checkpoint = state_now
                elif evaluation_coutdown < self.decrese_evaluation_limit:
                    if self.verbose >= 2:
                        print("{}:\t\tEpoch {} finished with {} validation error (worse {} times)".format(
                            self.name, epoch_number, evaluation_now, evaluation_now/evaluation_best))
                    evaluation_coutdown += 1
                else:
                    if self.verbose >= 2:
                        print("{}:\t\tEpoch {} finished with {} validation error (worse {} times) [END pretraining]".format(
                            self.name, epoch_number, evaluation_now, evaluation_now/evaluation_best))
                    elif self.verbose >= 1:
                        print("{}:\tEpoch {} finished [best evaluation: {}]".format(
                            self.name, epoch_number, evaluation_best))
                    break

                if self.maximum_epochs is not None and epoch_number >= self.maximum_epochs:
                    raise EpochNumTraining()

                try:
                    # init training data
                    dbn_dataset.set_up_train_data()
                    while True:
                        if self.maximum_time and (time.time() - initial_time)/60 > self.maximum_time:
                            raise TimeoutException()
                        _, global_step, *summaries_now = tf.get_default_session().run([self.training, self.global_step] + self.training_summaries)

                        if summary_obj is not None:
                            summary_obj.pass_summaries(global_step, *summaries_now)
                except tf.errors.OutOfRangeError:
                    epoch_number += 1

                if self.maximum_time is not None and (time.time() - initial_time)/60 > self.maximum_time:
                    raise TimeoutException()
        except TimeoutTraining:
            if self.verbose >= 1:
                print("{}:\tTraining finished: TIMEOUT [best evaluation: {}]".format(self.name, evaluation_best))
        except EpochNumTraining:
            if self.verbose >= 1:
                print("{}:\tTraining finished: EPOCH LIMIT [best evaluation: {}]".format(self.name, evaluation_best))
            
        self._load_last_checkpoint(evaluation_best_checkpoint)

    def _load_last_checkpoint(self, chekpoint_values):
        " load training "
        assert len(chekpoint_values) == len(self.checkpoint_placeholders)
        tf.get_default_session().run(self.checkpoint_operations,
            feed_dict={ph:v for ph,v in zip(self.checkpoint_placeholders, chekpoint_values)})






    def plot_filters(self, path, image_shape, colls=10, norm='max', scale=2):
        from sklearn.preprocessing import normalize
        W = tf.get_default_session().run(self.W)
        W = W[:, self.n_discriminative:]
        W = normalize(W, norm=norm, axis=1)
        self._plot_raw_data(W, path, image_shape, colls, scale=scale)

    def plot_generated_images(self, path, image_shape, number_of_images, colls=10, scale=2, prob=False):
        assert self.generative_gibbs_steps > 0
        images = self.generate_visible(number_of_images, prob=prob)
        images = images[:, self.n_discriminative:]
        self._plot_raw_data(images, path, image_shape, colls, scale=scale)

    def _plot_raw_data(self, data, path, image_shape, colls, padsize=2, scale=2):
        from PIL import Image
        images = [np.reshape(x,image_shape) for x in np.split(data, data.shape[0], axis=0)]
        out = None

        while len(images) > 0:
            i = images[:colls]
            del images[:colls]

            i = [np.pad(x, (padsize,padsize), 'constant') for x in i] 

            y = np.concatenate(i, axis=1)
            if out is None:
                out = y
            elif len(i) == colls:
                out = np.concatenate((out, y), axis=0)
            else:
                rem = image_shape[0]*colls + colls*padsize*2 - y.shape[1]
                y = np.concatenate((y, np.zeros((image_shape[1]+2*padsize, rem))), axis=1)
                out = np.concatenate((out, y), axis=0)

        out /= np.max(out)
        out *= 254
        out = out.astype(np.uint8)
        image = Image.fromarray(out, mode='L')
        image = image.resize((image.size[0]*scale, image.size[1]*scale), Image.NEAREST)
        image.save(path)






    @classmethod
    def _get_weight_variables_names(cls):
        """ for saving and loading purposes """
        # TODO  remove - use checkpoint system
        return ['W', 'b', 'c', 'persistant_chains', 'global_step', 'learning_rate']

    @classmethod
    def _get_param_names(cls):
        """ for saving and loading purposes """
        return ['n_discriminative',
                'n_binary_units',
                'n_normalized_gaussian_units',
                'n_persistant_chains',
                'contrastive_divergence_iter',
                'n_hidden_units',
                
                'initial_learning_rate',
                'learning_rate_decay',
                'minibatch_size',
                'momentum',
                'l1_regularization',
                'l2_regularization',
                'verbose']

    @classmethod
    def from_dict(cls, dct_to_load):
        """ loads RBM from dictionary """
        weights = {var_name: dct_to_load.pop(var_name) for var_name in cls._get_weight_variables_names()}

        n_visible_units = dct_to_load.pop('n_visible_units')

        instance = cls(**dct_to_load)
        setattr(instance, 'n_visible_units', n_visible_units)

        instance._build_model(weights)
        return instance



# **************************** # **************************** # **************************** #
# **************************** #           TESTS              # **************************** #
# **************************** # **************************** # **************************** #


if __name__ == '__main__':
    import unittest
    from sklearn import datasets
    import mydataset

    class TestRBN(unittest.TestCase):

        @classmethod
        def setUpClass(self):
            # data: [1797, 64], target: [1797,]
            (self.data, self.target) = datasets.load_digits(return_X_y=True)

        def _binarize_digits(self):
            return (self.data > 16/2).astype(np.float32)

        def _binarize_gaussian_digits(self):
            first = self.data[:,:32] > 16/2
            second = self.data[:,32:] / 16
            mean = np.mean(second, axis=0)
            std = np.std(second, axis=0)
            second = np.divide(second - mean, std, out=np.zeros_like(second), where=std!=0)
            #import pdb; pdb.set_trace();
            return np.concatenate((first, second), axis=1)

        def _create_digits_rbm(self, persistant=False, discriminative=False, gaussian=False, centering=False):
            binary = 64 if not gaussian else 32
            gaussian = 32 if gaussian else 0
            discriminative = 10 if discriminative else 0

            persistant = 77 if persistant else 0

            logged_losses_training=['mse', 'mse-split', 'reconstruction-split', 'reconstruction', 'free-energy', 'pseudo-likelihood', 'pseudo-likelihood-99']
            ls = ['mse']
            ls = logged_losses_training

            return BinaryRBM(
             n_binary_units=binary,
             n_normalized_gaussian_units=gaussian,
             n_discriminative=discriminative,
             n_persistant_chains=persistant,
             n_hidden_units=55,
             centering=centering,
             centering_decay=0.99,

             minibatch_size=100, learning_rate=1e-4, learning_rate_decay=0.999,
             contrastive_divergence_iter=3, 
             l1_regularization = 0.0001,
             l2_regularization = 0.0002,

             maximum_epochs=20, maximum_time=None,           
             target_loss='reconstruction', 
             decrese_evaluation_limit=5,
             logged_losses_training=ls,
             logged_losses_validation=ls,
             generative_gibbs_steps=10,
             verbose=2)

        def setUp(self):
            pass

        def test_build_generative(self):
            with TensorflowInitialize('build_generative'):
                dataset = mydataset.DBNDataset(nparray=self._binarize_digits())
                rbm = self._create_digits_rbm()
                rbm._build_model(dataset.get_iterator())

        def test_build_discriminative(self):
            with TensorflowInitialize('build_discriminative'):
                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True)
                rbm._build_model(dataset.get_iterator())

        def test_build_generative_persistant(self):
            with TensorflowInitialize('build_generative_persistant'):
                dataset = mydataset.DBNDataset(nparray=self._binarize_digits())
                rbm = self._create_digits_rbm(persistant=True)
                rbm._build_model(dataset.get_iterator())

        def test_build_discriminative_persistant(self):
            with TensorflowInitialize('build_discriminative_persistant'):
                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True)
                rbm._build_model(dataset.get_iterator())

        def test_build_discriminative_persistant_gaussian(self):
            with TensorflowInitialize('test_build_discriminative_persistant_gaussian'):
                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True, gaussian=True)
                rbm._build_model(dataset.get_iterator())

        def test_build_center(self):
            with TensorflowInitialize('test_build_center'):
                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True, gaussian=True, centering=True)
                rbm._build_model(dataset.get_iterator())

        '''
        def test_epoch_basic(self):
            with TensorflowInitialize():
                TensorflowInitialize().init_summary(subfolder='/build_test_epoch')

                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(),))
                rbm = self._create_digits_rbm()
                rbm.run_pretraining(dataset)

        def test_epoch_discriminative_persistant(self):
            with TensorflowInitialize():
                TensorflowInitialize().init_summary(subfolder='/build_test_epoch')

                dataset = mydataset.DBNDataset(nparray=(self._binarize_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True)
                rbm.run_pretraining(dataset)
        def test_epoch_persistant_gaussian(self):
            with TensorflowInitialize():
                TensorflowInitialize().init_summary(subfolder='/build_test_epoch')

                dataset = mydataset.DBNDataset(nparray=(self._binarize_gaussian_digits(),))
                rbm = self._create_digits_rbm(persistant=True, gaussian=True)
                rbm.run_pretraining(dataset)
        '''
        def test_binarize(self):
            data = self._binarize_gaussian_digits()
            self.assertFalse(np.any(np.isnan(data)))

        def test_epoch_discriminative_persistant_gaussian(self):
            with TensorflowInitialize('build_test_epoch_disc_pers_gauss') as s:
                dataset = mydataset.DBNDataset(nparray=(self._binarize_gaussian_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True, gaussian=True)
                rbm.run_pretraining(dataset, s)
                rbm.plot_filters('tests/filters.png', (8,8))
                rbm.plot_generated_images('tests/images.png', (8,8), 30)

        '''
        def test_plot_filters(self):
            with TensorflowInitialize('build_test_epoch_disc_pers_gauss') as s:
                dataset = mydataset.DBNDataset(nparray=(self._binarize_gaussian_digits(), self.target))
                rbm = self._create_digits_rbm(discriminative=True, persistant=True, gaussian=True)
                '''




        @unittest.skip("TODO")
        def test_save(self):
            #@unittest.skip("TODO")
            pass


        '''
        def test_pretrain_generative(self):
            pass
            #rbm = self._create_sup()
            #rbm.run_pretraining(self.data)
        '''



        


    unittest.main(verbosity=2)

    '''
    def _create_likelihood_cost(self):
        raise NotImplementedError
        """
        with tf.name_scope('Likelihood'):
            with tf.name_scope('CreateBitMask'):
                assert self.n_hidden_units < 30
                # TODO CONSTANT SHOUD BE INIT.
                initial_value = tf.expand_dims(tf.constant([False, True], dtype=tf.bool), axis=1)
                rem = self.n_hidden_units
                posible_values_of_h = tf.concat([
                        tf.reshape(tf.tile(initial_value, [2**(rem-m-1),2**(m)]), [2**rem, 1])
                        for m in range(rem)], axis=1)
                posible_values_of_h = tf.cast(posible_values_of_h, tf.float32)

            # TODO
            tf.reduce_mean(tf.log(pxn))
        """
        
    def _compute_hidden_units_matrix(self, matrix_visible_units):
        """
        Computes hidden unit outputs.
        :param matrix_visible_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return sess.run(self.compute_hidden_units_op,
                        feed_dict={self.visible_units_placeholder: matrix_visible_units})

    def _compute_visible_units_matrix(self, matrix_hidden_units):
        """
        Computes visible (or input) unit outputs.
        :param matrix_hidden_units: array-like, shape = (n_samples, n_features)
        :return:
        """
        return sess.run(self.compute_visible_units_op,
                        feed_dict={self.hidden_units_placeholder: matrix_hidden_units})
    '''

    '''
    self.Wd = None # discriminative weights
    self.bd = None # discriminative sample

    # DISCRIMINATIVE stuff
    if self.is_DISCRIMINATIVE:
        generator_Wb_sample = tf.distributions.Normal(loc=0., scale=0.01).sample(
                sample_shape=[self.n_hidden_units, self.discriminative])

        if label_sample is None:
            generator_bd_sample = tf.zeros(shape=[self.discriminative])
        else:
            generator_bd_sample = tf.constant(ProbTensorflowCalculator()(label_sample), dtype=tf.float32, shape=[self.discriminative])
            # 1/0 -> 0.
            generator_bd_sample = tf.where(tf.equal(generator_bd_sample, 1.), 
                    tf.zeros_like(generator_bd_sample), generator_bd_sample/(1.-generator_bd_sample))
            # log(0) -> 0.
            generator_bd_sample = tf.where(tf.equal(generator_bd_sample, 0.), 
                    tf.zeros_like(generator_bd_sample), tf.log(generator_bd_sample))

        self.Wd = tf.Variable(generator_Wb_sample, shape=[self.n_hidden_units, self.discriminative], dtype=tf.float32)
        self.bd = tf.Variable(generator_bd_sample, shape=[self.discriminative], dtype=tf.float32)
    '''
    '''
    def _stochastic_gradient_descent(self, _data):
        """
        Performs stochastic gradient descend optimization algorithm.
        :param _data: array-like, shape = (n_samples, n_features)
        :return:
        """
        for iteration in range(1, self.n_epochs + 1):

            idx = np.random.permutation(len(_data))
            data = _data[idx]
            for batch in batch_generator(self.minibatch_size, data):
                if len(batch) < self.minibatch_size:
                    # Pad with zeros
                    pad = np.zeros((self.minibatch_size - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
                    batch = np.vstack((batch, pad))
                self.sess.run([self.update_W, self.update_b, self.update_c],
                         feed_dict={self.visible_units_placeholder: batch})
            if self.verbose:
                error = self._compute_reconstruction_error(data)
                print(">> Epoch %d finished \tRBM Reconstruction error %f" % (iteration, error))
    '''
