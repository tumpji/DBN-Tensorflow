#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: mydataset.py
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
#      CREATED: 2019 03.13.
# =============================================================================

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DBNDataset:
    def __init__(self, 
            validation_proportion=0.15,        # split np.array into validation/train with this proportion 
            batch_size=100,                    #
            mapping=None,                      # function that can be used to map loaded values to something else
            random_state=42,                   # numpy array split initial random_state
            nparray=None,                      # data from np-array (this will be splitted)
            nparray_validation=None,
            nparray_train=None,
            prefetch_size=5
            ):
        ''' Create train and validation dataset and provide tf.data.iterator '''
        # datasets to create
        self.train_dataset = self.train_without_batch = self.validation_dataset = None
        # feed_dict for validation and test
        self.placeholder_dict_train = {}
        self.placeholder_dict_validation = {}
        # initializator for validation and tests
        self.v = self.t = None
        # iterator 
        self.iterator = None
        # conversion of input to features
        self.mapping = mapping

        self.prefetch_size = prefetch_size
        self.validation_proportion = validation_proportion
        self.batch_size = batch_size
        self.random_state = random_state

        # how many examples are loaded using provided array
        self.NPtrain = self.NPvalidation = 0
        
        if nparray is not None:
            assert nparray_train is None and nparray_validation is None
            self._load_numpy_array(data=self._convert_to_tuple(nparray))
        else:
            assert nparray_train is not None and nparray_validation is not None
            self._load_numpy_array(train=self._convert_to_tuple(nparray_train), validation=self._convert_to_tuple(nparray_validation))

        self._process_datasets()
    
    @staticmethod
    def _convert_to_tuple(obj):
        if isinstance(obj, (tuple, list)):
            return tuple(obj)
        else:
            return (obj,)
        

    def _process_datasets(self):
        """ shuffle map batch """
        self.train_dataset = self.train_dataset.shuffle(self.NPtrain)

        # 3) split it into batches
        self.train_without_batch = self.train_dataset.batch(self.NPtrain)
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.validation_dataset = self.validation_dataset.batch(self.NPvalidation)

        if self.mapping is not None:
            self.train_dataset = self.train_dataset.map(self.mapping)
            self.train_without_batch = self.train_without_batch.map(self.mapping)
            self.validation_dataset = self.validation_dataset.map(self.mapping)

        # 4) prefetch 
        self.train_dataset = self.train_dataset.prefetch(self.prefetch_size)

    def get_iterator(self):
        """ return train and validation iterator """ 
        if self.iterator is not None:
            return self.iterator
        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.t = self.iterator.make_initializer(self.train_dataset)
        self.t_without_batch = self.iterator.make_initializer(self.train_without_batch)
        self.v = self.iterator.make_initializer(self.validation_dataset)
        return self.iterator

    def set_up_train_data(self):
        tf.get_default_session().run(self.t, self.placeholder_dict_train)
    def set_up_train_without_minibatch_data(self):
        tf.get_default_session().run(self.t_without_batch, self.placeholder_dict_train)
    def set_up_validation_data(self):
        tf.get_default_session().run(self.v, self.placeholder_dict_validation)

    def _load_numpy_array(self, data=None, train=None, validation=None ):
        """ loads dataset from array, shuffle (no stratification) """
        assert data is None or isinstance(data, tuple)
        assert train is None or isinstance(train, tuple)
        assert validation is None or isinstance(validation, tuple)

        with tf.name_scope('loadNumpyArray'):
            # random split
            if data is not None:
                assert train is None and validation is None
                everything = train_test_split(*data, 
                        test_size=self.validation_proportion, random_state=self.random_state)

                train = everything[0::2]
                validation = everything[1::2]

            self.NPvalidation = validation[0].shape[0]
            self.NPtrain = train[0].shape[0]

            ph = tf.placeholder(tf.float32, shape=(None, *train[0].shape[1:]))
            if len(train) == 2:
                phlab = tf.placeholder(tf.int32, shape=(None, *train[1].shape[1:]))

                self.validation_dataset = tf.data.Dataset.from_tensor_slices((ph, phlab))
                self.train_dataset = tf.data.Dataset.from_tensor_slices((ph, phlab))

                self.placeholder_dict_validation[ph] = validation[0]
                self.placeholder_dict_validation[phlab] = validation[1]
                self.placeholder_dict_train[ph] = train[0]
                self.placeholder_dict_train[phlab] = train[1]

            elif len(train) == 1:
                self.validation_dataset = tf.data.Dataset.from_tensor_slices(ph)
                self.train_dataset = tf.data.Dataset.from_tensor_slices(ph)

                self.placeholder_dict_validation[ph] = validation[0]
                self.placeholder_dict_train[ph] = train[0]

