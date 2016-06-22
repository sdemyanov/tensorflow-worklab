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
#RESTORING_FILE = '/path/to/resnet-pretrained/ResNet-L101.ckpt'
RESTORING_FILE = None

EVAL_STEP_NUM = 100
DECAY_FACTOR = 0.1

#CHANGE
LEARNING_RATE = 0.01
EVAL_FREQUENCY = 500
STEP_VALUES = [20, 40]
LAST_STEP = STEP_VALUES[-1]


def learning_rate(step):
  lr = LEARNING_RATE
  for i in range(len(STEP_VALUES)):
    if (step > STEP_VALUES[i]):
      lr *= DECAY_FACTOR
    else:
      break
  return lr


def main(argv=None):
  writer = Writer(RESULTS_DIR)
  trainer = Trainer(RESULTS_DIR, 'train', writer)
  tester = Tester(RESULTS_DIR, 'valid', writer)

  step, _ = tester.test(EVAL_STEP_NUM)
  while (step < LAST_STEP):
    lr = learning_rate(step)
    step, _ = trainer.train(lr, EVAL_FREQUENCY, step, RESTORING_FILE)
    tester.test(EVAL_STEP_NUM, step)


if __name__ == '__main__':
  tf.app.run()