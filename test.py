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

import tester
reload(tester)
from tester import Tester

#import writer
#reload(writer)
#from writer import Writer

RESULTS_DIR = './current'
EVAL_STEP_NUM = 10

def main(argv=None):  # pylint: disable=unused-argument
  #writer = Writer(RESULTS_DIR)
  tester = Tester(RESULTS_DIR, 'test')#, writer)
  tester.test(EVAL_STEP_NUM)


if __name__ == '__main__':
  tf.app.run()