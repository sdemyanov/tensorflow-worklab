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

import tensorflow as tf

class Writer(object):

  def __init__(self, path):
    self._path = path
    print('Writer summary folder: %s' %self._path)
    if not tf.gfile.Exists(self._path):
      tf.gfile.MakeDirs(self._path)
    self._writer = tf.train.SummaryWriter(self._path)


  def write_graph(self, graph):
    self._writer.add_graph(graph)
    self._writer.flush()


  def write_summaries(self, summary_str, step):
    print('Writing string summaries to %s' %self._path)
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    self._writer.add_summary(summary, step)
    self._writer.flush()


  def write_scalars(self, dict, step):
    print('Writing dictionary summaries to %s' %self._path)
    summary = tf.Summary()
    for key, value in dict.items():
      summary.value.add(tag=key, simple_value=value)
    self._writer.add_summary(summary, step)
    self._writer.flush()