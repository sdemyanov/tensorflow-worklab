# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

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

import writer
reload(writer)
from writer import Writer

RESULTS_DIR = 'results'

EVAL_FREQUENCY = 500
LEARNING_RATE = 0.01

def main(argv=None):
  results_dir = os.path.join(dname, RESULTS_DIR)
  writer = Writer(results_dir)
  trainer = Trainer(results_dir, 'train', writer)
  trainer.train(LEARNING_RATE, EVAL_FREQUENCY)


if __name__ == '__main__':
  tf.app.run()