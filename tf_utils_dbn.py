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
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 03.21.
# =============================================================================
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

class SingletonMetaClass(type):
    instance = None
    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls.instance

class SingletonParameterMetaClass(type):
    instance = {}
    def __call__(cls, *args):
        try:
            out = cls.instance[args]
        except KeyError:
            out = cls.instance[args] = super(SingletonParameterMetaClass, cls).__call__(*args)
        return out





class TensorflowInitialize(metaclass=SingletonParameterMetaClass):
    """ Object for unique tensorflow session """
    LOGDIR = './logs'

    def __init__(self, subdirectory=""):
        self.subdirectory = '/' + self._append_subdirectory_date(subdirectory)
        self.filewriter = None
        self.new_graph()

    @staticmethod
    def _append_subdirectory_date(name):
        import datetime
        now = datetime.datetime.now()
        return name + now.strftime("%y-%m-%d_%H:%M:%S")

    def __enter__(self):
        self.sess = tf.Session(graph=self.graph).__enter__()
        #if self.filewriter is not None:
        #    self.filewriter.set_as_default()
            #self.filw = self.filewriter.as_default().__enter__()

        return self
    def __exit__(self, exc_type, exc_value, tb):
        #if self.filewriter is not None:
        #    self.filw.__exit__(self, exc_type, exc_value, tb)
        self.sess.__exit__(exc_type, exc_value, tb)

    def new_graph(self):
        self.graph = tf.Graph()
        return self.graph





    def init_summaries(self):
        """ after graph is build, then you shoud init summary """
        self.filewriter = tf.summary.FileWriter(self.LOGDIR+self.subdirectory, graph=self.graph)
        #self.filewriter = tf.contrib.summary.create_file_writer(self.LOGDIR+self.subdirectory)
        #with self.filewriter.as_default():
        #    tf.contrib.summary.always_record_summaries()

    def pass_summaries(self, global_step, *summaries):
        for summary in summaries:
            self.filewriter.add_summary(summary, global_step)


    def close_summaries(self):
        """ optional close of summary """
        if self.filewriter is not None:
            self._close_summary(self.filewriter)
            self.filewriter = None

    def add_scalar_summary(self, name, scalar, family=None):
        """ add scalar to output """
        s = tf.summary.scalar(name, scalar)
        #s = tf.contrib.summary.scalar(name, scalar, family=family)
        return s


    @staticmethod
    def _close_summary(summary):
        summary.close()


    '''
    def get_session(self):
        return self.sess
    '''
    '''
    @staticmethod
    def _close_session(sess):
        print('Closing session ...')
        sess.close()
    @classmethod
    def get_summaries(cls, *args, **kwargs):
        return cls.instance.get_summaries(*args, **kwargs)
    @classmethod
    def add_scalar_summary(cls, *args, **kwargs):
        return cls.instance.add_scalar_summary(*args, **kwargs)
    @classmethod
    def close_summaries(cls, *args, **kwargs):
        return cls.instance.close_summaries(*args, **kwargs)
    @classmethod
    def init_summary(cls, *args, **kwargs):
        return cls.instance.init_summary(*args, **kwargs)
    '''





    '''
    def merge_summaries(self):
        s = tf.summary.merge(self.unregistred_summaries)
        self.unregistred_summaries = list()
        return s
    '''
    






class BaseTensorFlowModel:
    def __init__(self):
        pass
        #self.sess = TensorflowInitialize().get_session()

    def save(self, save_path):
        import pickle

        with open(save_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @classmethod
    def load(cls, load_path):
        import pickle

        with open(load_path, 'rb') as fp:
            dct_to_load = pickle.load(fp)
            return cls.from_dict(dct_to_load)

    def to_dict(self):
        dct_to_save = {name: self.__getattribute__(name) for name in self._get_param_names()}
        dct_to_save.update(
            {name: self.__getattribute__(name).eval(sess) for name in self._get_weight_variables_names()})
        return dct_to_save

    @classmethod
    def from_dict(cls, dct_to_load):
        ''' load variables from dictionary'''
        pass

    def _build_model(self, weights=None):
        ''' builds tf model '''
        pass

    def _initialize_weights(self, weights):
        ''' load/initialize weights '''
        pass

    @classmethod
    def _get_param_names(cls):
        ''' get variable names to save '''
        pass




class AbstractSupervisedDBN:
    """
    Abstract class for supervised Deep Belief Network.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 unsupervised_dbn_class,
                 hidden_layers_structure=[100, 100],
                 activation_function='sigmoid',
                 optimization_algorithm='sgd',
                 learning_rate=1e-3,
                 learning_rate_rbm=1e-3,
                 n_iter_backprop=100,
                 l2_regularization=1.0,
                 n_epochs_rbm=10,
                 contrastive_divergence_iter=1,
                 batch_size=32,
                 dropout_p=0,  # float between 0 and 1. Fraction of the input units to drop
                 verbose=True):

        self.unsupervised_dbn = unsupervised_dbn_class(hidden_layers_structure=hidden_layers_structure,
                                                       activation_function=activation_function,
                                                       optimization_algorithm=optimization_algorithm,
                                                       learning_rate_rbm=learning_rate_rbm,
                                                       n_epochs_rbm=n_epochs_rbm,
                                                       contrastive_divergence_iter=contrastive_divergence_iter,
                                                       batch_size=batch_size,
                                                       verbose=verbose)

        self.unsupervised_dbn_class = unsupervised_dbn_class
        self.n_iter_backprop = n_iter_backprop
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.p = 1 - self.dropout_p
        self.verbose = verbose

    def fit(self, X, Y=None, pre_train=True, fine_tune=True):
        """
        Fits a model given data.
        :param X: array-like, shape = (n_samples, n_features)
        :param y : array-like, shape = (n_samples, )
        :param pre_train: bool
        :param fine_tune: bool
        :return:
        """
        if pre_train:
            self.pre_train(X, Y)
        if fine_tune:
            self._fine_tuning(X, Y)
        return self

    def predict(self, X):
        """
        Predicts the target given data.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        if len(X.shape) == 1:  # It is a single sample
            X = np.expand_dims(X, 0)
        transformed_data = self.transform(X)
        predicted_data = self._compute_output_units_matrix(transformed_data)
        return predicted_data

    def reversed_predict(self, Y):
        """
        Predicts the features given target data.
        :param Y: array-like, shape = (n_samples, n_classes)
        :return:
        """
        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, 0)

        raise NotImplementedError
        return predicted_data



    def pre_train(self, X, Y):
        """
        Apply unsupervised network pre-training.
        :param X: array-like, shape = (n_samples, n_features)
        :return:
        """
        self.unsupervised_dbn.fit(X, y=y)
        return self

    def fine_tune(self, X, Y):
        self._fine_tuning(X, Y)
        return self




    def transform(self, *args):
        return self.unsupervised_dbn.transform(*args)

    @abstractmethod
    def _transform_labels_to_network_format(self, labels):
        return

    @abstractmethod
    def _compute_output_units_matrix(self, matrix_visible_units):
        return

    @abstractmethod
    def _determine_num_output_neurons(self, labels):
        return

    @abstractmethod
    def _stochastic_gradient_descent(self, data, labels):
        return

    @abstractmethod
    def _fine_tuning(self, data, _labels):
        return

