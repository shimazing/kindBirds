"""Modification of https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import random
import numpy as np
import os
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from scipy.misc import logsumexp


class Replay(object):
  def __init__(self, batch_size, memory_size, observation_dims=[15]):
    self.batch_size = batch_size
    self.memory_size = memory_size
    # memory
    self.observations = np.empty([self.memory_size] + observation_dims,
                                dtype=np.float)
    self.birdtypes = np.empty([self.memory_size, 4], dtype=np.float)
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
            self.poststates, birds, terminals, None

  def sample_one(self):
    nonterminal = np.arange(self.count)[self.terminals[:self.count] == False]
    indices_ = nonterminal[nonterminal != ((self.current - 1) % self.count)]
    indices = (indices_ + 1) % self.count
    idx = np.random.choice(indices)

    prestate = self.observations[(idx - 1) % self.count]
    prebird = self.birdtypes[(idx - 1) % self.count]
    poststate = self.observations[idx]
    postbird = self.birdtypes[idx]
    action = self.actions[idx]
    reward = self.rewards[idx]
    terminal = self.terminals[idx]
    return prestate, prebird, action, reward, poststate,\
            postbird, terminal, 1


class PrioritizedReplay(object):
  def __init__(self, memory_size, observation_dims=[60, 105, 3], alpha=0.7, beta=0.5):
    # memory
    self.memory_size = memory_size
    self.prebirds = np.empty([self.memory_size, 4], dtype=np.float)
    self.postbirds = np.empty([self.memory_size, 4], dtype=np.float)
    self.actions = np.empty(self.memory_size, dtype=np.uint8)
    self.rewards = np.empty(self.memory_size, dtype=np.float)
    self.terminals = np.empty(self.memory_size, dtype=np.bool)
    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty([self.memory_size] + observation_dims, dtype = np.float16)
    self.poststates = np.empty([self.memory_size] + observation_dims, dtype = np.float16)

    self.alpha = alpha
    self.beta = beta
    self.priority_queue = {}
    self.max_priority = 1
    self.max_size = memory_size
    self.max_is_weight = 0
    self.size = 0

  def add(self, prestate, prebird, action,  reward, poststate, postbird, terminal):
    self.size += 1
    if self.size > self.max_size:
      self.size = self.max_size # replace the lowest priority experience

    e_id = self.size
    self.priority_queue[self.size] = [self.max_priority, e_id]
    self.up_heap(self.size)
    self.prestates[e_id, ...] = prestate
    self.poststates[e_id, ...] = poststate
    self.actions[e_id] = action
    self.rewards[e_id] = reward
    self.prebirds[e_id, ...] = prebird
    self.postbirds[e_id, ...] = postbird
    self.terminals[e_id] = terminal

  def up_heap(self, i):
    if i > 1:
      parent = math.floor(i /2)
      if self.priority_queue[parent][0] < self.priority_queue[i][0]:
        tmp = self.priority_queue[i]
        self.priority_queue[i] = self.priority_queue[parent]
        self.priority_queue[parent] = tmp
        # up heap parent
        self.up_heap(parent)

  def down_heap(self, i):
    if i < self.size:
      greatest = i
      left, right = i * 2, i * 2 + 1
      if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
        greatest = left
      if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
        greatest = right

      if greatest != i:
        tmp = self.priority_queue[i]
        self.priority_queue[i] = self.priority_queue[greatest]
        self.priority_queue[greatest] = tmp
        # down heap greatest
        self.down_heap(greatest)

  def get_priority(self):
    return list(map(lambda x: x[0], self.priority_queue.values()))
    #return list(map(lambda x: x[0], self.priority_queue.values()))[0:self.size]

  def sample_one(self, pred_net, target_net, discount, beta):
    self.beta = beta
    priority_list = self.get_priority()
    # sample transition
    prob_priority = np.array(priority_list, dtype=np.float64) ** self.alpha
    prob_priority = prob_priority / np.sum(prob_priority)
    selected_idx = np.random.multinomial(1,prob_priority)
    selected_idx = np.where(selected_idx == 1)[0][0]
    e_id = self.priority_queue[selected_idx+1][1]

    pj = priority_list[selected_idx]
    is_weight = (self.size * pj) ** self.beta
    if is_weight > self.max_is_weight:
      self.max_is_weight = is_weight
    is_weight = is_weight / self.max_is_weight

    prestate = self.prestates[e_id]
    prebird = self.prebirds[e_id]
    poststate = self.poststates[e_id]
    postbird = self.postbirds[e_id]
    action = self.actions[e_id]
    reward = self.rewards[e_id]
    terminal = self.terminals[e_id]

    if not terminal:
        max_action = pred_net.calc_eps_greedy_actions(poststate, postbird, 0)
        target = reward + \
            discount * target_net.calc_outputs_with_idx(
            np.expand_dims(poststate, axis=0),
            np.expand_dims(postbird, axis=0),
            [[0, max_action]])[0]
    else:
        target = reward

    update_priority = pred_net.get_priority(prestate, prebird, action, target) # no reference to pred_net, target
    if update_priority > self.max_priority:
      self.max_priority = update_priority
    self.priority_queue[selected_idx+1][0] = update_priority
    self.up_heap(selected_idx+1)
    self.down_heap(selected_idx+1)

    return prestate, prebird, action, reward, poststate,\
            postbird, terminal, is_weight, target



