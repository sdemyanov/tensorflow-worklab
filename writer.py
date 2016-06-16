# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:18:45 2016

@author: sdemyanov
"""

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