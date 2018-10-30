# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from builtins import object

import numpy as np
from abc import ABCMeta, abstractmethod


class BatchGeneratorBase(object):
    def __init__(self, data, BATCH_SIZE, num_batches=None, seed=False):
        __metaclass__ = ABCMeta
        self._data = data
        self.BATCH_SIZE = BATCH_SIZE
        self._num_batches = num_batches
        self._seed = seed
        self._resetted_rng = False
        self._iter_initialized = False
        self._p = None
        if self._num_batches is None:
            self._num_batches = 1e100
        self._batches_generated = 0

    def _initialize_iter(self):
        if self._seed is not False:
            np.random.seed(self._seed)
        self._iter_initialized = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self._iter_initialized:
            self._initialize_iter()
        if self._batches_generated >= self._num_batches:
            self._iter_initialized = False
            raise StopIteration
        minibatch = self.generate_train_batch()
        self._batches_generated += 1
        return minibatch

    @abstractmethod
    def generate_train_batch(self):
        '''override this'''
        '''
        Generate your batch from either self._train_data, self._validation_data or self._test_data. Make sure you
        generate the correct batch size (self.BATCH_SIZE)
        '''
        pass
