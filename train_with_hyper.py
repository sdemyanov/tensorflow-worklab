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

RESULTS_DIR = './results'
PARAMS_FILE = 'params.json'
HYPER_FILE = 'hyper.json'

#CHANGE
LEARNING_RATE = 0.001
EVAL_FREQUENCY = 500
EVAL_STEP_NUM = 100
PATIENCE = 3000
MAX_DECAYS = 0

DECAY_FACTOR = 0.1

#CHANGE
#RESTORING_FILE = '/path/to/resnet-pretrained/ResNet-L101.ckpt'
RESTORING_FILE = None
LAYERS_NUM = 25
HYPER_PATIENCE = 3 #layers

def model_file(step):
  return 'model.ckpt' + '-' + str(step)

def main(argv=None):

  hyper_file = os.path.join(RESULTS_DIR, HYPER_FILE)
  if (os.path.isfile(hyper_file)):
    with open(hyper_file, 'r') as handle:
      hyper = json.load(handle)
  else:
    hyper = {}
    hyper['min_test_step'] = LAYERS_NUM
    hyper['step'] = hyper['min_test_step']
    hyper['unchanged'] = 0
    hyper['restfile'] = RESTORING_FILE

  while (hyper['unchanged'] < HYPER_PATIENCE):
    results_dir = os.path.join(RESULTS_DIR, str(hyper['step']))
    writer = Writer(results_dir)
    trainer = Trainer(results_dir, 'train', writer, hyper['step'])
    tester = Tester(results_dir, 'valid', writer, hyper['step'])

    params_file = os.path.join(results_dir, PARAMS_FILE)
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

    if ('min_test_loss' not in hyper):
      hyper['min_test_loss'] = params['min_test_loss']

    while (params['num_decays'] <= MAX_DECAYS):
      params['step'], _ = trainer.train(params['learning_rate'], EVAL_FREQUENCY,
                                        params['step'], hyper['restfile'])
      _, test_loss = tester.test(EVAL_STEP_NUM, params['step'])
      if (test_loss < params['min_test_loss']):
        params['min_test_loss'] = test_loss
        params['min_test_step'] = params['step']
        params['unchanged'] = 0
      else:
        params['unchanged'] += EVAL_FREQUENCY
        if (params['unchanged'] >= PATIENCE):
          params['learning_rate'] *= DECAY_FACTOR
          params['num_decays'] += 1
          params['step'] = params['min_test_step']
          params['unchanged'] = 0

      with open(params_file, 'w') as handle:
        json.dump(params, handle, indent=2)
      print(params)

    #tester.test(step_num=None, init_step=params['min_test_step'])

    if (params['min_test_loss'] < hyper['min_test_loss']):
      hyper['min_test_loss'] = params['min_test_loss']
      hyper['min_test_step'] = hyper['step']
      hyper['unchanged'] = 0
    else:
      hyper['unchanged'] += 1

    hyper['restfile'] = os.path.join(results_dir, model_file(params['min_test_step']))
    hyper['step'] += 2
    with open(hyper_file, 'w') as handle:
      json.dump(hyper, handle, indent=2)
    print(hyper)

    print('\n NEW HYPER PARAMETER: %d' %hyper['step'])


if __name__ == '__main__':
  tf.app.run()