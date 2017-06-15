import tensorflow as tf
import numpy as np
from agent.agent import Agent
from environment.environment import Environment
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import pickle
import numpy as np
import pprint
import os
from collections import Counter
import  csv

flags = tf.app.flags

flags.DEFINE_boolean('is_train', True, '[True/False]')
flags.DEFINE_integer('n_batch', 100, 'batch size')
flags.DEFINE_integer('memory_size', 100000, '...')
flags.DEFINE_boolean('double_q', True, '...')
flags.DEFINE_float('discount', 1, '...')
flags.DEFINE_float('pretrain_steps', 0, '...')
flags.DEFINE_float('max_eps', 1, '...')
flags.DEFINE_float('min_eps', 0.1, '...')
flags.DEFINE_integer('annealing_steps', 500, '...')
flags.DEFINE_integer('update_freq', 10, '...')
flags.DEFINE_float('lr', 0.001, '...')
flags.DEFINE_integer('depth', 3, '...')
flags.DEFINE_string('hidden_sizes', '[]', '...')
flags.DEFINE_integer('seed', 123, '...')
flags.DEFINE_string('save_dir', 'savemodel', '...')
flags.DEFINE_integer('n_steps', 100000, '...')
conf = flags.FLAGS

tf.set_random_seed(conf.seed)
random.seed(conf.seed)

def main(*args, **kwargs):
    conf.save_dir = os.path.abspath(conf.save_dir)
    if not os.path.exists(conf.save_dir):
        os.makedirs(conf.save_dir)

    with tf.Session() as sess:
        env = Environment()
        agent = Agent(sess, conf, env, name='kindAgent')
        tf.global_variables_initializer().run()
        env.create()
        env.connect_client()
        if conf.is_train:
            agent.train()
        else:
            agent.competition()


if __name__ == '__main__':
  tf.app.run()
