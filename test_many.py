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
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append(dname)

import tester
reload(tester)
from tester import Tester

import writer
reload(writer)
from writer import Writer

RESULTS_DIR = 'results'

INTERVAL = 3
EVAL_STEP_NUM = 50

def main(argv=None):
  results_dir = os.path.join(dname, RESULTS_DIR)
  writer = Writer(results_dir)
  tester = Tester(results_dir, 'valid', writer)
  status_file = os.path.join(results_dir, 'checkpoint')
  last_update = 0
  while True:
    cur_time = os.stat(status_file).st_mtime
    if (cur_time > last_update):
      tester.test(EVAL_STEP_NUM)
      last_update = cur_time
    time.sleep(INTERVAL)


if __name__ == '__main__':
  tf.app.run()