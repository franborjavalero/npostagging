# Copyright (C) 2018  Fran-Borja Valero <franborjavalero@gmail.com>
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

class Rnn(object):
    
    def __init__(self, hidden, length_seq, bidirectional=False, dropout=None, unit_type='lstm_cpu', seed=8, name='rnn', training=False):
      
      self.hidden = hidden
      self.length_seq = length_seq
      self.bidirectional = bidirectional
      self.dropout = dropout
      self.unit_type = unit_type # lstm_cpu, gru_cpu, lstm_gpu, gru_gpu
      self.seed = seed
      self.name = name
      self.training = training

    @staticmethod
    def build_cell(hidden, batch_size, dropout=[], unit_type='gru_cpu', seed=8, training=False):

      if unit_type in ['lstm_cpu', 'lstm_gpu']:
        
        if unit_type == 'lstm_gpu':
          
          if training and len(dropout) > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(cell=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size), 
              output_keep_prob=prob) for num_layer, (size, prob) in enumerate(zip(hidden, dropout))]
          else:
            layers = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(size) for num_layer, size in enumerate(hidden)]

        else: # lstm_cpu

          if training and len(dropout) > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, name="layer_" + str(num_layer)), 
              output_keep_prob=prob) for num_layer, (size, prob) in enumerate(zip(hidden, dropout))]

          else:
            layers = [tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, name="layer_" + str(num_layer)) for num_layer, size in enumerate(hidden)]

        cells = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple=True)

        reset_state = cells.zero_state(batch_size, dtype=tf.float32)

      else:   # 'gru_cpu', 'gru_gpu'

        if unit_type == 'gru_gpu':

          if training and len(dropout) > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size),
              output_keep_prob=prob) for num_layer, (size, prob) in enumerate(zip(hidden, dropout))]

          else:
            layers = [tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(size) for num_layer, size in enumerate(hidden)]

        else: # 'gru_cpu'

          if training and len(dropout) > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.GRUCell(size, name="layer_" + str(num_layer)), 
              output_keep_prob=prob) for num_layer, (size, prob) in enumerate(zip(hidden, dropout))]

          else:
            layers = [tf.nn.rnn_cell.GRUCell(size, name="layer_" + str(num_layer)) for num_layer, size in enumerate(hidden)]

        cells = tf.nn.rnn_cell.MultiRNNCell(layers)

        zero_states = cells.zero_state(batch_size, dtype=tf.float32)

        reset_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) for state in zero_states])
      
      return cells, reset_state

    def forward(self, X):

      tf.set_random_seed(self.seed)
        
      with tf.variable_scope(self.name):

        batch_size = tf.shape(X)[0]

        with tf.name_scope("forward"):
          
          # rnn forward direction
          
          cells_fw, self.reset_state_fw = self.build_cell(self.hidden, batch_size, dropout=self.dropout, unit_type=self.unit_type,seed=self.seed, 
            training=self.training) 

          if self.bidirectional:

            with tf.name_scope("backward"):

              # rnn backward direction

              cells_bw, self.reset_state_bw = self.build_cell(self.hidden, batch_size, dropout=self.dropout, unit_type=self.unit_type, seed=self.seed, 
                training=self.training)
                
              self.bi_outputs, self.bi_states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw, cell_bw=cells_bw, sequence_length=self.length_seq, inputs=X, 
                initial_state_fw=self.reset_state_fw, initial_state_bw=self.reset_state_bw, scope="{}_bidirectional".format(self.name))

              self.output = tf.concat(self.bi_outputs, 2, name="{}_concatenated_output".format(self.name))

              self.state = tf.contrib.layers.fully_connected(tf.concat(self.bi_states, 1), self.hidden[-1])
        
          else:
            self.output, self.state = tf.nn.dynamic_rnn(cells_fw, X, self.length_seq, self.reset_state_fw, scope="{}_unidirectional".format(self.name))

  