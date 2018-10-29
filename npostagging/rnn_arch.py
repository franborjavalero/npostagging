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

import os
import sys
import math
import time
import datetime
import numpy as np
import npostagging.rnn
import npostagging.data
import tensorflow as tf

class RnnArchitecture(object):
    
  def __init__(self, model_name, hidden, num_amb, num_tags, bidirectional=False, dropout=[], learning_rate=0.003, training=False, unit_type='lstm_cpu', 
    seed=8, max_grad_norm=5):

    self.model_name = model_name
    self.training = training
    self.seed = seed
    self.num_amb = num_amb
    self.num_tags = num_tags
    self.max_grad_norm = max_grad_norm
    self.learning_rate = learning_rate
    self.hidden = hidden
    self.bidirectional = bidirectional
    self.dropout = dropout
    self.unit_type = unit_type
    self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="gstep")
    
    self.seq_X = tf.placeholder(dtype=tf.int32, shape=[None, None], name="X")
    self.seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="seq_length")
    self.seq_y = tf.placeholder(dtype=tf.int32, shape=[None, None], name="y")
    
    self.mask_unk = tf.placeholder(dtype=tf.float32, shape=[None, None], name="mask_unk")
    self.mask_ambiguous = tf.placeholder(dtype=tf.float32, shape=[None, None], name="mask_ambiguous")
    
    self.amb_multihot = tf.placeholder(dtype=tf.float32, shape=[num_amb, num_tags], name="ambiguity_multihot")

  def forward(self):

    tf.set_random_seed(self.seed)
    
    X = tf.nn.embedding_lookup(self.amb_multihot, self.seq_X, name="X_multihot")

    self.rnn = npostagging.rnn.Rnn(self.hidden, self.seq_length, bidirectional=self.bidirectional, dropout=self.dropout, unit_type=self.unit_type, 
      seed=self.seed, name='rnn')
    
    self.rnn.forward(X)
    
    self.logits = tf.layers.dense(inputs=self.rnn.output, units=self.num_tags, activation=None, name='logits')    
    mask_lengths = tf.sequence_mask(self.seq_length, dtype=tf.float32, name="mask_lengths")
    
    self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.seq_y, mask_lengths, name="loss")
    self.cost = tf.reduce_mean(self.loss, name="cost")
    
    if self.training:
      
      with tf.variable_scope("fit"):
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm) 
        self.opt = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables), global_step=self.gstep)

    with tf.variable_scope("evaluate"):
      
      softmax_outputs = tf.nn.softmax(self.logits, name="softmax_outputs")
      predictions = tf.cast(tf.argmax(softmax_outputs, axis=2), tf.int32, name="predictions")
      correct_predictions = tf.equal(self.seq_y, predictions, name="correct_predictions")
      correct_predictions_flatten = tf.reshape(correct_predictions, [-1], name="correct_predictions_flatten")
      mask_lengths_flatten = tf.reshape(mask_lengths, [-1], name="mask_lengths_flatten")
      mask_unk_flatten = tf.reshape(self.mask_unk, [-1], name="mask_unk_flatten")
      mask_ambiguous_flatten = tf.reshape(self.mask_ambiguous, [-1], name="mask_ambiguous_flatten")
      predictions_flatten = tf.reshape(predictions, [-1], name="predictions_flatten")
        
      self.acc_total = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_flatten, mask_lengths_flatten), tf.float32), name="total_accuracy")
      self.acc_ambiguous = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_flatten, mask_ambiguous_flatten), tf.float32), name="ambiguous_accuracy")
      self.acc_unk = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_flatten, mask_unk_flatten), tf.float32), name="unk_accuracy")
        
      self.tags = tf.boolean_mask(predictions_flatten, mask_lengths_flatten, name="tags")

  def evaluate(self, filename_X, filename_y, matrix_multihot, ids_ambiguous, id_unk_token):
    
    data_input = npostagging.data.Data(filename_X, self.num_amb-1) # self.num_amb-1 is <pad>
    data_output = npostagging.data.Data(filename_y, self.num_tags-1)

    saver = tf.train.Saver()

    checkpoint_name = os.path.join('models', self.model_name, "checkpoint")

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_name))
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
      
      sess.run(data_input.iterator.initializer)
      sess.run(data_output.iterator.initializer)

      current_X, current_X_lengths = sess.run(data_input.next_element)
      current_y_tag, _ = sess.run(data_output.next_element)

      mask_unk = (current_X == id_unk_token)
      mask_ambiguous = np.array([ids_ambiguous[current_X[i][j]] for i in range(current_X.shape[0]) for j in range(current_X.shape[1])])
      mask_ambiguous = mask_ambiguous.reshape(mask_unk.shape)
      
      cost_tag, acc_total_tag, acc_ambiguous_tag, acc_unk_tag = sess.run([self.cost, self.acc_total, self.acc_ambiguous, self.acc_unk],
        feed_dict={self.seq_X:current_X, self.amb_multihot:matrix_multihot, self.seq_length:current_X_lengths, self.mask_unk:mask_unk, 
        self.mask_ambiguous:mask_ambiguous, self.seq_y:current_y_tag})
      
      print("Tag - cost: {:.3f} acc_total: {:.3f}, acc_ambiguous: {:.3f}, acc_unk: {:.3f}".format(cost_tag, acc_total_tag, acc_ambiguous_tag, acc_unk_tag))

  def predict(self, filename_input, matrix_multihot):

    data_input = npostagging.data.Data(filename_input, self.num_amb-1)

    saver = tf.train.Saver()

    checkpoint_name = os.path.join('models', self.model_name, "checkpoint")

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_name))
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
      
      sess.run(data_input.iterator.initializer)

      current_X, current_X_lengths = sess.run(data_input.next_element)

      predictions = sess.run([self.tags], feed_dict={self.seq_X:current_X, self.amb_multihot:matrix_multihot, self.seq_length:current_X_lengths})

      return predictions

# train rnn_arch with early stopping

def train_rnn_arch(arch_desc, filename_X_train, filename_y_train, filename_X_dev, filename_y_dev, num_epochs, batch_size, matrix_multihot, ids_ambiguous, id_unk_token, num_ambiguities, 
  num_tags, max_stopping_step=5):
    
  min_cost_train = sys.float_info.max
  min_cost_dev = sys.float_info.max

  stopping_step = 0
  num_batches = None  
  
  ckpt_aux = os.path.join('models', arch_desc['model_name']+"_aux", "checkpoint")
  ckpt_final = os.path.join('models', arch_desc['model_name'], "checkpoint")
  
  time_init_train = time.time()
  
  for idx_epoch in range(num_epochs):

    time_init_epoch = time.time()
      
    train_graph = tf.Graph()
    
    with train_graph.as_default():
      
      input_train = npostagging.data.Data(filename_X_train, num_ambiguities-1, batch_size=batch_size) # num_ambiguities-1 is <pad>
      output_train = npostagging.data.Data(filename_y_train, num_tags-1, batch_size=batch_size) # num_tags-1 is <pad>
      
      rnn_arch_train = npostagging.rnn_arch.RnnArchitecture(arch_desc['model_name']+"_aux", arch_desc['hidden_rnn'], num_ambiguities, num_tags, 
        bidirectional=arch_desc["bidirectional_rnn"], dropout=arch_desc["dropout_rnn"], learning_rate=arch_desc["learning_rate"], 
			  training=True, unit_type=arch_desc["unit_type"], seed=arch_desc["seed"], max_grad_norm=arch_desc["max_grad_norm"])
      
      rnn_arch_train.forward()

      saver = tf.train.Saver()
      
      with tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=train_graph) as sess_train:
        
        sess_train.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_aux))
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess_train, ckpt.model_checkpoint_path)
        
        sess_train.run(input_train.iterator.initializer)
        sess_train.run(output_train.iterator.initializer)
        
        num_batches = math.ceil(input_train.num_sentences / input_train.batch_size)
        
        for idx_batch_train in range(num_batches):
          
          time_init_batch = time.time()
          
          current_X_train, current_X_lengths_train = sess_train.run(input_train.next_element)
          current_y_train, _ = sess_train.run(output_train.next_element)
          
          mask_ambiguous = np.array([ids_ambiguous[current_X_train[i][j]] for i in range(current_X_train.shape[0]) for j in range(current_X_train.shape[1])])
          mask_ambiguous = mask_ambiguous.reshape(current_X_train.shape)
          
          cost_train, total_acc_train, ambiguous_acc_train, _ = sess_train.run([rnn_arch_train.cost, rnn_arch_train.acc_total, rnn_arch_train.acc_ambiguous, 
            rnn_arch_train.opt], feed_dict={rnn_arch_train.seq_X:current_X_train, rnn_arch_train.amb_multihot:matrix_multihot, 
            rnn_arch_train.seq_length:current_X_lengths_train, rnn_arch_train.seq_y:current_y_train, rnn_arch_train.mask_ambiguous:mask_ambiguous})
            
          time_end_batch = time.time()
          time_batch = datetime.timedelta(seconds=round(time_end_batch-time_init_batch))

          print("Epoch {}, Batch {}, cost: {:.3f}, acc_total: {:.3f}, acc_ambiguous: {:.3f}, time {} - train".format(idx_epoch, idx_batch_train, cost_train,
            total_acc_train, ambiguous_acc_train, str(time_batch)))

          step = idx_epoch*num_batches + idx_batch_train
          
          if round(cost_train,3) < round(min_cost_train,3):
            saver.save(sess_train, ckpt_aux, step)
            min_cost_train = cost_train
        
        time_end_epoch = time.time()
        time_epoch = datetime.timedelta(seconds=round(time_end_epoch-time_init_epoch))
        print("Epoch {}, time {} - train".format(idx_epoch, str(time_epoch)))
    
    sess_train.close()

    dev_graph = tf.Graph()
    
    with dev_graph.as_default():
      
      input_dev = npostagging.data.Data(filename_X_dev, num_ambiguities-1)
      output_dev = npostagging.data.Data(filename_y_dev, num_tags-1)
      
      rnn_arch_dev = npostagging.rnn_arch.RnnArchitecture(arch_desc['model_name'], arch_desc['hidden_rnn'], num_ambiguities, num_tags, bidirectional=arch_desc["bidirectional_rnn"], 
        unit_type=arch_desc["unit_type"])

      rnn_arch_dev.forward()
      
      saver = tf.train.Saver()
      
      with tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=dev_graph) as sess_dev:
        
        sess_dev.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_aux))
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess_dev, ckpt.model_checkpoint_path)
          
        sess_dev.run(input_dev.iterator.initializer)
        sess_dev.run(output_dev.iterator.initializer)

        current_X, current_X_lengths = sess_dev.run(input_dev.next_element)
        current_y_tag, _ = sess_dev.run(output_dev.next_element)

        mask_unk = (current_X == id_unk_token)
        mask_ambiguous = np.array([ids_ambiguous[current_X[i][j]] for i in range(current_X.shape[0]) for j in range(current_X.shape[1])])
        mask_ambiguous = mask_ambiguous.reshape(mask_unk.shape)
        
        cost_dev, total_acc_dev, ambiguous_acc_dev, unk_acc_dev = sess_dev.run([rnn_arch_dev.cost, rnn_arch_dev.acc_total, rnn_arch_dev.acc_ambiguous, 
          rnn_arch_dev.acc_unk], feed_dict={rnn_arch_dev.seq_X:current_X, rnn_arch_dev.amb_multihot:matrix_multihot, rnn_arch_dev.seq_length:current_X_lengths, 
            rnn_arch_dev.mask_unk:mask_unk, rnn_arch_dev.mask_ambiguous:mask_ambiguous, rnn_arch_dev.seq_y:current_y_tag})
        
        print("Epoch {} cost: {:.3f}, acc_total: {:.3f}, acc_ambiguous: {:.3f}, acc_unk: {:.3f} - dev".format(idx_epoch, cost_dev, total_acc_dev, ambiguous_acc_dev, unk_acc_dev))
      
        # early stopping

        if round(cost_dev,3) < round(min_cost_dev,3):
          saver.save(sess_dev, ckpt_final, idx_epoch)
          min_cost_dev = cost_dev
          stopping_step = 0

        else:
          stopping_step += 1

    sess_dev.close()
    
    if stopping_step >= max_stopping_step: 
      print("Early stopping is trigger at Epoch {} - dev".format(idx_epoch))
      break

  time_end_train = time.time()
  time_train = datetime.timedelta(seconds=round(time_end_train-time_init_train))
  print("Total time {} - train".format(str(time_train)))