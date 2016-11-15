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
from shutil import copyfile

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append(dname)

from classes.trainer import Trainer
from classes.writer import Writer
import paths

GPU = 1
TRAIN_DECAY = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EVAL_FREQUENCY = 100

TRAIN_INIT = {'is_train': True,
              'gpu': GPU,
              'decay': TRAIN_DECAY,
              'batch_size': BATCH_SIZE,
              'fold_name': paths.TRAIN_FOLD,
              'results_dir': paths.RESULTS_DIR,
              'write_graph': False}

TRAIN_PARAMS = {'restoring_file': paths.RESTORING_FILE,
                'init_step': None,
                'step_num': EVAL_FREQUENCY,
                'learning_rate': LEARNING_RATE,
                'momentum': MOMENTUM,
                'print_frequency': 10,
                'save_frequency': None,
                'model_name': paths.MODEL_NAME}

def main(argv=None):
  MODEL_FILE = 'mnist_classifier.py'
  copyfile(os.path.join(dname, 'classes', MODEL_FILE), 
          os.path.join(paths.RESULTS_DIR, MODEL_FILE))

  writer = Writer(paths.RESULTS_DIR) # Writer summary folder
  trainer = Trainer(TRAIN_INIT, writer)
  trainer.train(TRAIN_PARAMS)


if __name__ == '__main__':
  tf.app.run()