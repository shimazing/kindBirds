"""Modification of https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import random
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Replay(object):
  def __init__(self, batch_size, memory_size, observation_dims=[15]):
    self.batch_size = batch_size
    self.memory_size = memory_size
    # memory
    self.observations = np.empty([self.memory_size] + observation_dims,
                                dtype=np.float)
    self.birdtypes = np.empty([self.memory_size, 3], dtype=np.float)
    self.actions = np.empty(self.memory_size, dtype=np.uint8)
    self.rewards = np.empty(self.memory_size, dtype=np.float)
    self.terminals = np.empty(self.memory_size, dtype=np.bool)
    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty([self.batch_size] + observation_dims, dtype = np.float16)
    self.poststates = np.empty([self.batch_size] + observation_dims, dtype = np.float16)

    self.count = 0
    self.current = 0

  def add(self, observed, bird, reward, action, terminal):
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    if observed is not None:
        self.observations[self.current, ...] = observed
        self.birdtypes[self.current, ...] = bird
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1) # How many data?
    print(self.count, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    self.current = (self.current + 1) % self.memory_size # Current position

  def sample(self):
    # indices = []
    # TODO double check if it is work
    nonterminal = np.arange(self.count)[self.terminals[:self.count] == False]
    indices_ = nonterminal[nonterminal != ((self.current - 1) % self.count)]
    indices = (indices_ + 1) % self.count
    '''
    while len(indices) < self.batch_size:
      while True:
        index = random.randint(0, self.count - 1)
        #index = random.randint(1, self.count - 1)
        if index == self.current: #
            continue
        if self.terminals[(index - 1) % self.count]:
            continue
        break

      self.prestates[len(indices), ...] = self.retreive(index - 1)
      self.poststates[len(indices), ...] = self.retreive(index)
      indices.append(index)
    '''
    self.prestates = self.observations[indices_]
    self.poststates = self.observations[indices]
    #indices = np.array(indices)
    actions = self.actions[indices]
    rewards = self.rewards[indices]
    terminals = self.terminals[indices]
    prebirds = self.birdtypes[indices_]
    birds = self.birdtypes[indices]
    return self.prestates, prebirds, actions, rewards, \
            self.poststates, birds, terminals

