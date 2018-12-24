# Copyright (C) 2018 Francisco de Borja Valero <franborjavalero@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf

class Nn(object):
    
    def __init__(self, hidden, dropout=[], seed=8, name='nn', training=False):
      
      self.hidden_sizes = hidden
      self.dropout = dropout
      self.seed = seed
      self.name = name
      self.training = training
      
    def forward(self, X):
              
      tf.set_random_seed(self.seed)
      
      with tf.variable_scope(self.name):

        out = tf.layers.dense(inputs=X, units=self.hidden_sizes[0], activation=tf.nn.relu, name="nn_layer0")

        if len(self.dropout) > 0:
          out = tf.layers.dropout(inputs=out, rate=self.dropout[0], seed=self.seed, training=self.training, name="nn_layer0_dropout")

        for id_ in range(len(self.hidden_sizes[1:])):

          out = tf.layers.dense(inputs=out, units=self.hidden_sizes[id_+1], activation=tf.nn.relu, name="nn_layer{}".format(id_+1))

          if len(self.dropout) > 0:
            out = tf.layers.dropout(inputs=out, rate=self.dropout[id_+1], seed=self.seed, training=self.training, name="nn_layer{}_dropout".format(id_+1))

        self.output = out