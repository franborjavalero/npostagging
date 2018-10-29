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
import npostagging.nn
import npostagging.rnn
import npostagging.data
import tensorflow as tf

class TwoPhasesArchitecture(object):

  def __init__(self, model_name, hidden_rnn, hidden_nn, num_amb, num_tags, bidirectional=False, dropout_rnn=[], dropout_nn=[], learning_rate=0.003, 
    training=False, unit_type='lstm_cpu', seed=8, max_grad_norm=5):
    
    self.model_name = model_name
    self.num_amb = num_amb
    self.num_tags = num_tags
    self.seed = seed
    self.learning_rate = learning_rate
    self.training = training
    self.max_grad_norm = max_grad_norm
    self.hidden_rnn = hidden_rnn
    self.dropout_rnn = dropout_rnn
    self.bidirectional = bidirectional
    self.hidden_nn = hidden_nn
    self.dropout_nn = dropout_nn
    self.unit_type = unit_type
    self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="gstep")

    self.seq_X = tf.placeholder(tf.int32, [None, None], name="X")
    self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="seq_lenght")
    self.seq_y_tag = tf.placeholder(tf.int32, [None, None], name="y_tag")
    self.seq_y_amb = tf.placeholder(tf.int32, [None, None], name="y_next_ambiguity")
    
    self.mask_unk = tf.placeholder(tf.float32, [None, None], name="mask_unk")
    self.mask_ambiguous = tf.placeholder(tf.float32, [None, None], name="mask_ambiguous")
    
    self.amb_multihot = tf.placeholder(dtype=tf.float32, shape=[num_amb, num_tags], name="ambiguity_multihot")

  def forward(self):

    tf.set_random_seed(self.seed)

    # phase 1: predict next ambiguity class
    
    with tf.variable_scope("phase1"):

      X_amb = tf.one_hot(self.seq_X, self.num_amb, name="X_onehot_ambiguity")

      self.rnn = npostagging.rnn.Rnn(self.hidden_rnn, self.seq_lengths, bidirectional=self.bidirectional, dropout=self.dropout_rnn, unit_type=self.unit_type, 
        seed=self.seed,name='rnn', training=self.training)

      self.rnn.forward(X_amb)
        
      self.logits_amb = tf.layers.dense(inputs=self.rnn.output, units=self.num_amb, activation=None, name='logits_next_ambiguity')
      mask_lengths = tf.sequence_mask(self.seq_lengths, dtype=tf.float32, name="mask_lengths")
      
      self.loss_amb = tf.contrib.seq2seq.sequence_loss(self.logits_amb, self.seq_y_amb, mask_lengths, name="loss_next_ambiguity")
      self.cost_amb = tf.reduce_mean(self.loss_amb, name="cost_next_ambiguity")

      if self.training:

        with tf.variable_scope("fit"):
          optimizer = tf.train.AdamOptimizer(self.learning_rate)
          trainable_variables = tf.trainable_variables()
          gradients = tf.gradients(self.loss_amb, trainable_variables)
          clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm) 
          self.opt_amb = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables), global_step=self.gstep)

      with tf.variable_scope("evaluate"):
        softmax_outputs_amb = tf.nn.softmax(self.logits_amb, name="softmax_outputs_next_ambiguity")
        predictions_amb = tf.cast(tf.argmax(softmax_outputs_amb, axis=2), tf.int32, name="predictions_next_ambiguity")
        correct_predictions = tf.equal(self.seq_y_amb, predictions_amb, name="correct_predictions_next_ambiguity")
        correct_predictions_flatten = tf.reshape(correct_predictions, [-1], name="correct_predictions_next_ambiguity_flatten")
        mask_lengths_flatten = tf.reshape(mask_lengths, [-1], name="mask_lengths_flatten")
        
        self.acc_amb = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_flatten, mask_lengths_flatten), tf.float32), name="accuracy_next_ambiguity")

    # phase 2: predict current tag

    with tf.name_scope("phase2"):

      self.nn = npostagging.nn.Nn(self.hidden_nn, dropout=self.dropout_nn, seed=self.seed, name='nn', training=self.training)

      self.nn.forward(self.rnn.output)
    
      self.logits_tag = tf.layers.dense(inputs=self.nn.output, units=self.num_tags, activation=None, name="logits_tag")

      self.loss_tag = tf.contrib.seq2seq.sequence_loss(self.logits_tag, self.seq_y_tag, mask_lengths, name="loss_tag")
      self.cost_tag = tf.reduce_mean(self.loss_tag, name="cost_tag")

      if self.training:

        with tf.name_scope("fit"):
          optimizer = tf.train.AdamOptimizer(self.learning_rate)
          trainable_variables = tf.trainable_variables()
          gradients = tf.gradients(self.loss_tag, trainable_variables)
          clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm) 
          self.opt_tag = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables), global_step=self.gstep)
    
      with tf.name_scope("evaluate"):

        softmax_logits_tag = tf.nn.softmax(self.logits_tag, name="softmax_logits_tag")        
        predictions_tag = tf.cast(tf.argmax(softmax_logits_tag, axis=2), tf.int32, name="predictions_tag")
        correct_predictions_tag = tf.equal(self.seq_y_tag, predictions_tag, name="correct_predictions_tag")
        correct_predictions_tag_flatten = tf.reshape(correct_predictions_tag, [-1], name="correct_predictions_tag_flatten")
        mask_lengths_flatten = tf.reshape(mask_lengths, [-1], name="mask_lengths_flatten")
        mask_unk_flatten = tf.reshape(self.mask_unk, [-1], name="mask_unk_flatten")
        mask_ambiguous_flatten = tf.reshape(self.mask_ambiguous, [-1], name="mask_ambiguous_flatten")
        
        self.acc_total_tag = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_tag_flatten, mask_lengths_flatten), tf.float32), name="total_accuracy_tag")
        self.acc_unk_tag = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_tag_flatten, mask_unk_flatten), tf.float32), name="unk_accuracy_tag")
        self.acc_ambiguous_tag = tf.reduce_mean(tf.cast(tf.boolean_mask(correct_predictions_tag_flatten, mask_ambiguous_flatten), tf.float32), name="ambiguous_accuracy_tag")
        
        predictions_tag_flatten = tf.reshape(predictions_tag, [-1], name="predictions_tag_flatten")
        self.tags = tf.boolean_mask(predictions_tag_flatten, mask_lengths_flatten, name="tags")
  
  def evaluate(self, filename_X, filename_y_amb, filename_y_tag, matrix_multihot, ids_ambiguous, id_unk_token):

    data_input = npostagging.data.Data(filename_X, self.num_amb-1, filename2=filename_y_amb)
    data_input2 = npostagging.data.Data(filename_y_tag, self.num_tags-1)

    saver = tf.train.Saver()

    checkpoint_name = os.path.join('models', self.model_name, "checkpoint")

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_name))
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
      
      sess.run(data_input.iterator.initializer)
      sess.run(data_input2.iterator.initializer)

      ((current_X, current_X_lengths), (current_y_amb, _)) = sess.run(data_input.next_element)
      current_y_tag, _ = sess.run(data_input2.next_element)

      mask_unk = (current_X == id_unk_token)
      mask_ambiguous = np.array([ids_ambiguous[current_X[i][j]] for i in range(current_X.shape[0]) for j in range(current_X.shape[1])])
      mask_ambiguous = mask_ambiguous.reshape(mask_unk.shape)

      cost_amb, acc_amb, cost_tag, acc_total_tag, acc_ambiguous_tag, acc_unk_tag = \
        sess.run([self.cost_amb, self.acc_amb, self.cost_tag, self.acc_total_tag, self.acc_ambiguous_tag, self.acc_unk_tag],
          feed_dict={self.seq_X:current_X, self.seq_y_amb:current_y_amb, self.seq_y_tag:current_y_tag, self.seq_lengths:current_X_lengths,
            self.mask_ambiguous:mask_ambiguous, self.mask_unk:mask_unk, self.amb_multihot:matrix_multihot})

      print("Next ambiguity task - cost: {:.3f}, acc: {:.3f}, \nTag task - cost: {:.3f}, total_acc: {:.3f}, ambiguous_acc: {:.3f}, unk_acc: {:.3f}".format(cost_amb, 
        acc_amb, cost_tag, acc_total_tag, acc_ambiguous_tag, acc_unk_tag))

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

      predictions = sess.run([self.tags], feed_dict={self.seq_X:current_X, self.seq_lengths:current_X_lengths, self.amb_multihot:matrix_multihot})

      return predictions

# train phase 1 with early stopping

def train_rnn_nn_arch_phase1(arch_desc, filename_X_train, filename_y_train, filename_X_dev, filename_y_dev, num_epochs, batch_size, num_ambiguities, num_tags, max_stopping_step=5):
    
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
      
      input_train = npostagging.data.Data(filename_X_train, num_ambiguities-1, filename2=filename_y_train, batch_size=batch_size)
      
      rnn_nn_arch_train = TwoPhasesArchitecture(arch_desc['model_name']+"_aux", arch_desc['hidden_rnn'], arch_desc["hidden_nn"], num_ambiguities, 
			num_tags, bidirectional=arch_desc["bidirectional_rnn"], dropout_rnn=arch_desc["dropout_rnn"], dropout_nn=arch_desc["dropout_nn"], 
			learning_rate=arch_desc["learning_rate"], training=True, unit_type=arch_desc["unit_type"], seed=arch_desc["seed"], 
			max_grad_norm=arch_desc["max_grad_norm"])
      
      rnn_nn_arch_train.forward()

      saver = tf.train.Saver()
      
      with tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=train_graph) as sess_train:
        
        sess_train.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_aux))
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess_train, ckpt.model_checkpoint_path)
        
        sess_train.run(input_train.iterator.initializer)
        
        num_batches = math.ceil(input_train.num_sentences / input_train.batch_size)
        
        for idx_batch_train in range(num_batches):
          
          time_init_batch = time.time()
          
          ((current_X_train, current_X_lengths_train), (current_y_train, _)) = sess_train.run(input_train.next_element)
          
          cost_train, acc_train, _ = sess_train.run([rnn_nn_arch_train.cost_amb, rnn_nn_arch_train.acc_amb, rnn_nn_arch_train.opt_amb], 
            feed_dict={rnn_nn_arch_train.seq_X:current_X_train, rnn_nn_arch_train.seq_lengths:current_X_lengths_train, rnn_nn_arch_train.seq_y_amb:current_y_train})
            
          time_end_batch = time.time()
          time_batch = datetime.timedelta(seconds=round(time_end_batch-time_init_batch))
            
          print("Phase 1- Epoch {}, Batch {}, cost: {:.3f}, acc: {:.3f}, time {} - train".format(idx_epoch, idx_batch_train, cost_train, acc_train, str(time_batch)))
            
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
      
      input_dev = npostagging.data.Data(filename_X_dev, num_ambiguities-1, filename2=filename_y_dev)
      
      rnn_nn_arch_dev = TwoPhasesArchitecture(arch_desc['model_name']+"_aux", arch_desc['hidden_rnn'], arch_desc["hidden_nn"], num_ambiguities, 
        num_tags, bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

      rnn_nn_arch_dev.forward()
      
      saver = tf.train.Saver()
      
      with tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=dev_graph) as sess_dev:
        
        sess_dev.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_aux))
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess_dev, ckpt.model_checkpoint_path)
          
        sess_dev.run(input_dev.iterator.initializer)

        ((current_X, current_X_lengths), (current_y_tag, _)) = sess_dev.run(input_dev.next_element)
        
        cost_dev, acc_dev = sess_dev.run([rnn_nn_arch_dev.cost_amb, rnn_nn_arch_dev.acc_amb], feed_dict={rnn_nn_arch_dev.seq_X:current_X, 
          rnn_nn_arch_dev.seq_lengths:current_X_lengths, rnn_nn_arch_dev.seq_y_amb:current_y_tag})
        
        print("Phase 1- Epoch {} cost: {:.3f}, acc: {:.3f} - dev".format(idx_epoch, cost_dev, acc_dev))
      
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
  
# train phase 2 with early stopping

def train_rnn_nn_arch_phase2(arch_desc, filename_X_train, filename_y_train, filename_X_dev, filename_y_dev, num_epochs, batch_size, matrix_multihot, ids_ambiguous, id_unk_token, 
  num_ambiguities, num_tags, max_stopping_step=5):
    
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
      
      input_train = npostagging.data.Data(filename_X_train, num_ambiguities-1, batch_size=batch_size)
      output_train = npostagging.data.Data(filename_y_train, num_tags-1, batch_size=batch_size)
      
      rnn_nn_arch_train = TwoPhasesArchitecture(arch_desc['model_name']+"_aux", arch_desc['hidden_rnn'], arch_desc["hidden_nn"], num_ambiguities, 
			  num_tags, bidirectional=arch_desc["bidirectional_rnn"], dropout_rnn=arch_desc["dropout_rnn"], dropout_nn=arch_desc["dropout_nn"], 
			  learning_rate=arch_desc["learning_rate"], training=True, unit_type=arch_desc["unit_type"], seed=arch_desc["seed"], 
			  max_grad_norm=arch_desc["max_grad_norm"])
      
      rnn_nn_arch_train.forward()

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
          
          cost_train, total_acc_train, ambiguous_acc_train, _ = sess_train.run([rnn_nn_arch_train.cost_tag, rnn_nn_arch_train.acc_total_tag, 
            rnn_nn_arch_train.acc_ambiguous_tag, rnn_nn_arch_train.opt_tag], feed_dict={rnn_nn_arch_train.seq_X:current_X_train, 
            rnn_nn_arch_train.amb_multihot:matrix_multihot, rnn_nn_arch_train.seq_lengths:current_X_lengths_train, rnn_nn_arch_train.seq_y_tag:current_y_train, 
            rnn_nn_arch_train.mask_ambiguous:mask_ambiguous})
            
          time_end_batch = time.time()
          time_batch = datetime.timedelta(seconds=round(time_end_batch-time_init_batch))
            
          print("Phase 2 - Epoch {}, Batch {}, cost: {:.3f}, acc_total: {:.3f}, acc_ambiguous: {:.3f}, time {} - train".format(idx_epoch, idx_batch_train, cost_train, 
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
      
      rnn_nn_arch_dev = TwoPhasesArchitecture(arch_desc['model_name']+"_aux", arch_desc['hidden_rnn'], arch_desc["hidden_nn"], num_ambiguities, 
        num_tags, bidirectional=arch_desc["bidirectional_rnn"], unit_type=arch_desc["unit_type"])

      rnn_nn_arch_dev.forward()
      
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
        
        cost_dev, total_acc_dev, ambiguous_acc_dev, unk_acc_dev = sess_dev.run([rnn_nn_arch_dev.cost_tag, rnn_nn_arch_dev.acc_total_tag, 
          rnn_nn_arch_dev.acc_ambiguous_tag, rnn_nn_arch_dev.acc_unk_tag], feed_dict={rnn_nn_arch_dev.seq_X:current_X, rnn_nn_arch_dev.amb_multihot:matrix_multihot, 
          rnn_nn_arch_dev.seq_lengths:current_X_lengths, rnn_nn_arch_dev.mask_unk:mask_unk, rnn_nn_arch_dev.mask_ambiguous:mask_ambiguous, 
          rnn_nn_arch_dev.seq_y_tag:current_y_tag})
        
        print("Phase 2 - Epoch {} cost: {:.3f}, acc_total: {:.3f}, acc_ambiguous: {:.3f}, acc_unk: {:.3f} - dev".format(idx_epoch, cost_dev, total_acc_dev, ambiguous_acc_dev, 
          unk_acc_dev))
      
        # early stopping

        if round(cost_dev,3) < round(min_cost_dev,3):
          saver.save(sess_dev, ckpt_final, idx_epoch)
          min_cost_dev = cost_dev
          stopping_step = 0
          last_epoch = idx_epoch

        else:
          stopping_step += 1
    
    sess_dev.close()
    
    if stopping_step >= max_stopping_step: 
      print("Early stopping is trigger at Epoch {} - dev".format(idx_epoch))
      break

  time_end_train = time.time()
  time_train = datetime.timedelta(seconds=round(time_end_train-time_init_train))
  print("Total time {} - train".format(str(time_train)))