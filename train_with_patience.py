#  Copyright 2016-present Sergey Demyanov. All Rights Reserved.
#
#  Contact: my_name@my_sirname.net
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import json

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append(dname)

from classes.trainer import Trainer
from classes.tester import Tester
from classes.writer import Writer

import paths

#CHANGE
GPU = 0
TRAIN_DECAY = 0.99
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EVAL_FREQUENCY = 1000
EVAL_STEP_NUM = 625
PATIENCE = 3000 / EVAL_FREQUENCY
MAX_DECAYS = 2
DECAY_FACTOR = 0.1
VALID_FOLD = paths.VALID_FOLD
VALID_FOLD = paths.TEST_FOLD

TRAIN_INIT = {'is_train': True,
              'gpu': GPU,
              'decay': TRAIN_DECAY,
              'batch_size': TRAIN_BATCH_SIZE,
              'fold_name': paths.TRAIN_FOLD,
              'results_dir': paths.RESULTS_DIR}

TEST_INIT = {'is_train': False,
             'gpu': GPU,
             'decay': TRAIN_DECAY,
             'batch_size': TEST_BATCH_SIZE,
             'fold_name': VALID_FOLD,
             'results_dir': paths.RESULTS_DIR}

TRAIN_PARAMS = {'restoring_file': paths.RESTORING_FILE,
                'init_step': None,
                'step_num': EVAL_FREQUENCY,
                'learning_rate': LEARNING_RATE,
                'momentum': MOMENTUM,
                'print_frequency': 10,
                'save_frequency': None}

TEST_PARAMS = {'restoring_file': None,
               'init_step': None,
               'step_num': EVAL_STEP_NUM,
               'epoch_num': 1,
               'load_results': False}


def main(argv=None):

  writer = Writer(paths.RESULTS_DIR)
  trainer = Trainer(TRAIN_INIT, writer)

  tester = Tester(TEST_INIT, writer)

  if os.path.isfile(paths.PARAMS_FILE):
    with open(paths.PARAMS_FILE, 'r') as handle:
      params = json.load(handle)
  else:
    params = TRAIN_PARAMS
    params['min_test_step'], params['min_test_loss'] = tester.test(TEST_PARAMS)
    params['init_step'] = params['min_test_step']
    params['unchanged'] = 0
    params['num_decays'] = 0

  while (params['num_decays'] <= MAX_DECAYS):
    params['init_step'], _ = trainer.train(params)
    _, test_loss = tester.test(TEST_PARAMS)
    if (test_loss < params['min_test_loss']):
      params['min_test_loss'] = test_loss
      params['min_test_step'] = params['init_step']
      params['unchanged'] = 0
    else:
      params['unchanged'] += 1
      if (params['unchanged'] >= PATIENCE):
        params['learning_rate'] *= DECAY_FACTOR
        params['num_decays'] += 1
        params['init_step'] = params['min_test_step']
        params['unchanged'] = 0

    with open(paths.PARAMS_FILE, 'w') as handle:
      json.dump(params, handle, indent=2)
    print(params)

  #tester.test(step_num=None, init_step=params['min_test_step'])


if __name__ == '__main__':
  tf.app.run()