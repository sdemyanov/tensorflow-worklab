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

#CHANGE
RESULTS_DIR = './results'
PARAMS_FILE = 'params.json'
RESTORING_FILE = None
#RESTORING_FILE = '/path/to/resnet-pretrained/ResNet-L50.ckpt'
RESTORING_FILE = '/data/mma/skin/molemap/resnet-pretrained/ResNet-L50.ckpt'

#CHANGE
LEARNING_RATE = 0.01
EVAL_FREQUENCY = 500
EVAL_STEP_NUM = 100
PATIENCE = 3000 / EVAL_FREQUENCY
MAX_DECAYS = 2

DECAY_FACTOR = 0.1

def main(argv=None):

  writer = Writer(RESULTS_DIR)
  trainer = Trainer(RESULTS_DIR, 'train', writer)
  tester = Tester(RESULTS_DIR, 'valid', writer)

  params_file = os.path.join(RESULTS_DIR, PARAMS_FILE)
  if (os.path.isfile(params_file)):
    with open(params_file, 'r') as handle:
      params = json.load(handle)
  else:
    params = {}
    params['min_test_step'], params['min_test_loss'] = tester.test(EVAL_STEP_NUM)
    params['step'] = params['min_test_step']
    params['unchanged'] = 0
    params['num_decays'] = 0
    params['learning_rate'] = LEARNING_RATE

  while (params['num_decays'] <= MAX_DECAYS):
    params['step'], _ = trainer.train(params['learning_rate'], EVAL_FREQUENCY,
                                      params['step'], RESTORING_FILE)
    _, test_loss = tester.test(EVAL_STEP_NUM, params['step'])
    if (test_loss < params['min_test_loss']):
      params['min_test_loss'] = test_loss
      params['min_test_step'] = params['step']
      params['unchanged'] = 0
    else:
      params['unchanged'] += 1
      if (params['unchanged'] >= PATIENCE):
        params['learning_rate'] *= DECAY_FACTOR
        params['num_decays'] += 1
        params['step'] = params['min_test_step']
        params['unchanged'] = 0

    with open(params_file, 'w') as handle:
      json.dump(params, handle, indent=2)
    print(params)

  #tester.test(step_num=None, init_step=params['min_test_step'])


if __name__ == '__main__':
  tf.app.run()