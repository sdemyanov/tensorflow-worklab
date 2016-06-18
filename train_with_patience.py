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

import trainer
reload(trainer)
from trainer import Trainer

import tester
reload(tester)
from tester import Tester

import writer
reload(writer)
from writer import Writer

RESULTS_DIR = './current'
PARAMS_FILE = 'params'

#CHANGE
LEARNING_RATE = 0.01
EVAL_FREQUENCY = 600
EVAL_STEP_NUM = 100
PATIENCE = 3000
MAX_DECAYS = 4

DECAY_FACTOR = 0.1

def main(argv=None):

  writer = Writer(RESULTS_DIR)
  trainer = Trainer(RESULTS_DIR, 'train', writer)
  tester = Tester(RESULTS_DIR, 'valid', writer)

  params_file = os.path.join(RESULTS_DIR, PARAMS_FILE)
  if (os.path.isfile(params_file)):
    with open(params_file, 'r') as handle:
      params = json.load(handle)
    min_test_loss = params['min_test_loss']
    min_test_step = params['min_test_step']
    unchanged = params['unchanged']
    num_decays = params['num_decays']
    step = params['step']
    learning_rate = params['learning_rate']
  else:
    min_test_step, min_test_loss = tester.test(EVAL_STEP_NUM)
    unchanged = 0
    num_decays = 0
    step = min_test_step
    learning_rate = LEARNING_RATE

  while (num_decays <= MAX_DECAYS):
    step, _ = trainer.train(learning_rate, EVAL_FREQUENCY, step)
    _, test_loss = tester.test(EVAL_STEP_NUM, step)
    if (test_loss < min_test_loss):
      min_test_loss = test_loss
      min_test_step = step
      unchanged = 0
    else:
      unchanged += EVAL_FREQUENCY
      if (unchanged >= PATIENCE):
        learning_rate *= DECAY_FACTOR
        num_decays += 1
        step = min_test_step
        unchanged = 0

    params = {'min_test_loss': min_test_loss,
              'min_test_step': min_test_step,
              'unchanged': unchanged,
              'num_decays': num_decays,
              'step': step,
              'learning_rate': learning_rate}
    with open(params_file, 'w') as handle:
      json.dump(params, handle)

  print('min_test_loss: %.3f' %min_test_loss)
  print('min_test_step: %d' %min_test_step)
  tester.test(step_num=None, init_step=min_test_step)


if __name__ == '__main__':
  tf.app.run()