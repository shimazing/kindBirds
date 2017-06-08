import tensorflow as tf
import numpy as np
import math
from environment.environment import Environment
from .replay import Replay
from layer.cnn import CNN
import os
import sys

class Agent(object):
    def __init__(self, sess, conf, env, name='agent'):
        self.sess = sess
        self.n_steps = conf.n_steps
        self.name = name

        self.save_dir = os.path.relpath(conf.save_dir)
        assert os.path.exists(self.save_dir)
        self.save_path = os.path.join(self.save_dir, self.name + '.ckpt')

        self.define_action_space()
        self.n_actions = self.n_angle * self.n_taptime

        self.n_batch = conf.n_batch
        self.memory_size = conf.memory_size
        self.double_q = conf.double_q
        self.discount = conf.discount
        self.pretrain_steps = conf.pretrain_steps
        self.max_lr = 0.01
        self.min_lr = 0.01
        self.lr = conf.lr
        self.update_freq = conf.update_freq
        try:
            self.depth = conf.depth
        except AttributeError:
            self.depth = 3 # number of channels (3 RGB channel)

        self.observation_dims = [105, 60] + [self.depth]

        self.states = tf.placeholder('float32',
                [None] + self.observation_dims, name='states')
        self.birdtypes = tf.placeholder('float32',
                [None, 3], name='slingbirdtype')

        # build controllers
        self.build_Qnet()
        # env setting
        self.env = env
        # replay memory
        self.replay = Replay(self.n_batch, self.memory_size,
                self.observation_dims)

    def build_Qnet(self):

        self.pred_net = CNN(
            self.sess,
            None,
            self.n_actions,
            inputs=self.states,
            birdtypes=self.birdtypes,
            name='pred')

        self.target_net = CNN(
            self.sess,
            None,
            self.n_actions,
            inputs=self.states,
            birdtypes=self.birdtypes,
            name='target')
        self.target_net.create_copy_op(self.pred_net)

    def define_action_space(self):
        angles = np.arange(0.025, 0.5, 0.025) * math.pi
        self.n_angle = len(angles)
        taptimes = np.arange(0, 4000, 200)
        self.n_taptime = len(taptimes)
        self.action_space = []
        for angle in angles:
            for taptime in taptimes:
                self.action_space.append((angle, taptime))
        print(">> Total action space size : ", len(self.action_space))

    def train(self):
        self.restore_model()
        self.target_net.run_copy()
        total_steps = 0
        eps = 0.1
        for epi in range(self.n_steps):
            print("BBBBBBBBBBBBBBBBBBBBBBB")
            terminal = False
            action = -1
            reward = 0
            state = None
            prev_state = state
            birdtype = np.zeros(3)
            prev_birdtype = birdtype
            while not terminal:
                terminal, reward, state, birdtype_, n_birds = \
                        self.env.get_state_reward()
                birdtype = np.zeros(3)
                if birdtype_ is not None:
                    birdtype[birdtype_] = 1
                # TODO : replay memory edit for birdtype
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                self.replay.add(state, birdtype, reward, action, terminal) # save state, terminal or not and reward after action
                # add(prev_state, action, reward, state, terminal)
                if not terminal:
                    prev_state = state
                    prev_birdtype = birdtype
                    action = self.pred_net.calc_eps_greedy_actions(state, \
                            birdtype, eps)
                    # action idx -> theta, v
                    angle, taptime = self.action_space[action]
                    self.env.act(angle, taptime)
            prestates, prebirds, actions, rewards, poststates, postbirds, terminals = self.replay.sample()

            if self.double_q:
                max_actions = self.pred_net.calc_actions(poststates, postbirds)
                targets = self.target_net.calc_outputs_with_idx(poststates,
                    [[idx, pred_a] for idx, pred_a in enumerate(max_actions)])
            else:
                targets = self.target_net.calc_max_outputs(poststates, postbirds)
            targets = targets * np.where(terminals, 0, 1)
            targets = rewards + self.discount * targets
            self.pred_net.optimize(prestates, prebirds, actions, targets, lr)

            if epi % self.update_freq == self.update_freq - 1:
                self.target_net.run_copy()

        # TODO : save every n steps
        saver = tf.train.saver()
        saver.save(self.sess, self.save_path)

    def test(self):
        self.restore_model()
        pass

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.save_dir, ckpt_name)
            saver.restore(self.sess, fname)
            print("[*] Load SUCCESS: %s" % self.save_path)
            return True
        else:
            print("[!] Load FAILED: %s" % self.save_path)
            return False

