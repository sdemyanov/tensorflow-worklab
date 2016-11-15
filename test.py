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

import tensorflow as tf
import os
import json
from shutil import copyfile

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append(dname)

from classes.tester import Tester

import paths

#CHANGE
GPU = 0
TRAIN_DECAY = 0.99
BATCH_SIZE = 16
TEST_FOLD = paths.TEST_FOLD
#TEST_FOLD = paths.VALID_FOLD
EVAL_STEP_NUM = None
#EVAL_STEP_NUM = int(32 / Tester.BATCH_SIZE)
LOAD_RESULTS = False
#LOAD_RESULTS = True

TEST_INIT = {'is_train': False,
             'gpu': GPU,
             'decay': TRAIN_DECAY,
             'batch_size': BATCH_SIZE,
             'fold_name': TEST_FOLD,
             'results_dir': paths.RESULTS_DIR}

TEST_PARAMS = {'restoring_file': paths.RESTORING_FILE,
               'init_step': None,
               'step_num': EVAL_STEP_NUM,
               'epoch_num': 1,
               'load_results': LOAD_RESULTS}

def main(argv=None):

  if os.path.isfile(paths.PARAMS_FILE):
    with open(paths.PARAMS_FILE, 'r') as handle:
      params = json.load(handle)
      TEST_PARAMS['init_step'] = params['min_test_step']
      print('init_step: %d' % TEST_PARAMS['init_step'])
  else:
    print('init_step is None')

  tester = Tester(TEST_INIT)
  tester.test(TEST_PARAMS)


if __name__ == '__main__':
  tf.app.run()