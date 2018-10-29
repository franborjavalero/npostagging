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

import math
import npostagging.utils
import tensorflow as tf

# tf.data class for datasets with variable sequence lengths

class Data(object):

  @staticmethod
  def load_dataset_from_text_file(filename):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(lambda string: tf.string_split([string]).values) # sequence values
    dataset = dataset.map(lambda tokens: (tf.string_to_number(tokens, tf.int32), tf.size(tokens))) # sequence length
    return dataset

  def __init__(self, filename1, id_pad_token, batch_size=None, num_epochs=1, filename2=None, seed=8):

    # by defeault bach_size = number sentences (samples) of corpus: test and dev mode
    # by defeault num_epochs = 1 : test and dev mode
    
    self.num_sentences = npostagging.utils.get_num_sentences(filename1)
    self.num_epochs = num_epochs

    if batch_size:
      self.batch_size = batch_size
    else:
      self.batch_size = self.num_sentences
      
    if filename2:

      dataset = tf.data.Dataset.zip((self.load_dataset_from_text_file(filename1), self.load_dataset_from_text_file(filename2))) # (filename1, filename2)
      padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))) # ((tensor, scalar),(tensor, scalar))          
      padding_values = ((id_pad_token, 0), (id_pad_token, 0)) # ((sequence_tokens, sequence_length), (sequence_tokens, sequence_length))

      if self.num_epochs > 1:
        dataset = (dataset
          .shuffle(buffer_size=self.num_sentences, seed=seed, reshuffle_each_iteration=True)
          .padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
          .repeat(self.num_epochs))
      else: 
        dataset = (dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values))

    else:

      dataset = self.load_dataset_from_text_file(filename1) # (filename1)
      padded_shapes = (tf.TensorShape([None]), tf.TensorShape([])) # (tensor, scalar)         
      padding_values = (id_pad_token, 0) # (sequence_tokens, sequence_length)
      
      if self.num_epochs > 1:
        dataset = (dataset
          .shuffle(buffer_size=self.num_sentences, seed=seed, reshuffle_each_iteration=True)
          .padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
          .repeat(self.num_epochs))
      else:
        dataset = (dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values))

    self.iterator = dataset.make_initializable_iterator()
    self.next_element = self.iterator.get_next()
