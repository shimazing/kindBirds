import tensorflow as tf
import random
from .layer import *
import numpy as np

class Network(object):
  def __init__(self, sess, name):
    self.sess = sess
    self.copy_op = None
    self.name = name
    self.var = {}
    self.lr = tf.placeholder(tf.float32, shape=[])

  def build_output_ops(self, input_layer, network_output_type,
      value_hidden_sizes, advantage_hidden_sizes, output_size,
      weights_initializer, biases_initializer, hidden_activation_fn,
      output_activation_fn, trainable):

    self.output_size = output_size
    if network_output_type == 'normal':
      self.outputs, self.var['w_out'], self.var['b_out'] = \
          linear(input_layer, output_size, weights_initializer,
                 biases_initializer, output_activation_fn, trainable, name='out')
    elif network_output_type == 'dueling':
      # Dueling Network
      assert len(value_hidden_sizes) != 0 and len(advantage_hidden_sizes) != 0

      layer = input_layer
      for idx, hidden_size in enumerate(value_hidden_sizes):
        w_name, b_name = 'val_w_%d' % idx, 'val_b_%d' % idx

        layer, self.var[w_name], self.var[b_name] = \
            linear(layer, hidden_size, weights_initializer,
              biases_initializer, hidden_activation_fn, trainable, name='val_lin_%d' % idx)

      self.value, self.var['val_w_out'], self.var['val_w_b'] = \
          linear(layer, 1, weights_initializer,
            biases_initializer, output_activation_fn, trainable, name='val_lin_out')

      layer = input_layer
      for idx, hidden_size in enumerate(advantage_hidden_sizes):
        w_name, b_name = 'adv_w_%d' % idx, 'adv_b_%d' % idx

        layer, self.var[w_name], self.var[b_name] = \
            linear(layer, hidden_size, weights_initializer,
              biases_initializer, hidden_activation_fn, trainable, name='adv_lin_%d' % idx)

      self.advantage, self.var['adv_w_out'], self.var['adv_w_b'] = \
          linear(layer, output_size, weights_initializer,
            biases_initializer, output_activation_fn, trainable, name='adv_lin_out')

      # Simple Dueling
      # self.outputs = self.value + self.advantage

      # Max Dueling
      # self.outputs = self.value + (self.advantage -
      #     tf.reduce_max(self.advantage, reduction_indices=1, keep_dims=True))

      # Average Dueling
      self.outputs = self.value + (self.advantage -
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))

    self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)
    self.outputs_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
    self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)
    self.actions = tf.argmax(self.outputs, dimension=1)

  def run_copy(self):
    if self.copy_op is None:
      raise Exception("run `create_copy_op` first before copy")
    else:
      self.sess.run(self.copy_op)

  def create_copy_op(self, network):
    with tf.variable_scope(self.name):
      copy_ops = []

      for name in self.var.keys():
        copy_op = self.var[name].assign(network.var[name])
        copy_ops.append(copy_op)

      self.copy_op = tf.group(*copy_ops, name='copy_op')

  def calc_actions(self, observation, birdtypes):
      return self.actions.eval({self.inputs: observation,
          self.birdtypes: birdtypes}, session=self.sess)

  def calc_outputs(self, observation, birdtypes):
      return self.outputs.eval({self.inputs: observation,
          self.birdtypes: birdtypes}, session=self.sess)

  def calc_max_outputs(self, observation, birdtypes):
    return self.max_outputs.eval({self.inputs: observation,
        self.birdtypes: birdtypes}, session=self.sess)

  def calc_outputs_with_idx(self, observation, birdtypes, idx):
    return self.outputs_with_idx.eval(
            {self.inputs: observation, self.birdtypes: birdtypes,
                self.outputs_idx: idx}, session=self.sess)

  def calc_eps_greedy_actions(self, observation, birdtype, eps=0):
    shape = [1] + list(observation.shape)
    assert len(shape) == 4
    if random.random() < eps:
        return random.randint(0, self.output_size - 1)
    else:
        observation = observation.reshape(shape)
        birdtype = birdtype.reshape(1, -1)
        return self.calc_actions(observation, birdtype)[0]

  def build_train_op(self):
    self.targets = tf.placeholder('float32', [None], name='target_q_t')
    #self.actions = tf.placeholder('int64', [None], name='action')
    #actions_one_hot = tf.one_hot(self.actions, self.n_actions)
    pred_q = self.outputs_with_idx
    #tf.reduce_sum(self.outputs * actions_one_hot,
    #                       reduction_indices=1,
    #                       name='q_acted')
    delta = self.targets - pred_q
    self.priority = tf.abs(delta)
    self.loss = tf.reduce_mean(tf.square(delta))
    optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
    self.optim = optimizer.minimize(self.loss)

  def optimize(self, inputs, birdtypes, actions, targets, lr):
    outputs_idx = [[idx, pred_a] for idx, pred_a in enumerate(actions)]
    self.sess.run(self.optim, feed_dict={
        self.inputs: inputs,
        self.birdtypes: birdtypes,
        self.outputs_idx: outputs_idx,
        self.targets: targets,
        self.lr: lr})

  def get_priority(self, input, birdtype, action, target):
    outputs_idx = [[0, action]]
    inputs = np.expand_dims(input, axis=0)
    birdtypes = np.expand_dims(birdtype, axis=0)
    targets = np.expand_dims(target, axis=0)
    return self.sess.run(self.priority, feed_dict={
        self.inputs: inputs,
        self.birdtypes: birdtypes,
        self.outputs_idx: outputs_idx,
        self.targets: targets})[0]
