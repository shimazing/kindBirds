import tensorflow as tf
import numpy as np
import math
from .replay import Replay, PrioritizedReplay
from layer.cnn import CNN
import os
import sys
import pickle
import traceback

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
        self.eps = conf.max_eps
        self.annealing_steps = conf.annealing_steps
        self.min_eps = conf.min_eps
        self.step = (conf.max_eps - conf.min_eps) / conf.annealing_steps
        self.max_lr = 0.01
        self.min_lr = 0.01
        self.lr = conf.lr
        self.update_freq = conf.update_freq
        self.beta = 0.5
        self.beta_step = - 0.5 / conf.annealing_steps

        try:
            self.depth = conf.depth
        except AttributeError:
            self.depth = 3 # number of channels (3 RGB channel)

        self.observation_dims = [60, 105] + [self.depth]

        self.states = tf.placeholder('float32',
                [None] + self.observation_dims, name='states')
        self.birdtypes = tf.placeholder('float32',
                [None, 4], name='slingbirdtype')

        # build controllers
        self.build_Qnet()
        # env setting
        self.env = env
        # replay memory
        self.replay = PrioritizedReplay(self.memory_size,
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
        angles = np.arange(0.05, 0.4, 0.027) * math.pi
        self.n_angle = len(angles)
        taptimes = np.arange(0, 4000, 300)
        self.n_taptime = len(taptimes)
        self.action_space = []
        for angle in angles:
            for taptime in taptimes:
                self.action_space.append((angle, taptime))
        print(">> Total action space size : ", len(self.action_space))

    def train_replay(self):
      try:
        self.restore_model()
        if os.path.exists(os.path.join(self.save_dir, 'replay.p')):
            with open(os.path.join(self.save_dir, 'replay.p'), 'rb') as file:
                try:
                    replay = pickle.load(file)
                    self.replay = replay
                except:
                    print("No replay")
                    return
                    pass
                print(self.replay.size, "CURRENT MEMORY SIZE")
        self.target_net.run_copy()
        step = 0
        print("OPTIMIZE")

        for j in range(10000):
            prestates = []
            prebirds = []
            actions = []
            rewards = []
            poststates = []
            postbirds = []
            terminals = []
            weights = []
            targets = []

            for i in range(self.n_batch):
                prestate, prebird, action_, reward_, poststate, postbird,\
                        terminal_, weight, target = self.replay.sample_one(
                                self.pred_net,
                                self.target_net,
                                self.discount,
                                self.beta)
                if reward_ <= 0:
                    reward_ = -1
                prestates.append(prestate)
                prebirds.append(prebird)
                actions.append(action_)
                rewards.append(reward_)
                poststates.append(poststate)
                postbirds.append(postbird)
                terminals.append(terminal_)
                weights.append(weight)
                targets.append(target)

            prestates = np.stack(prestates, axis=0)
            prebirds = np.stack(prebirds, axis=0)
            actions = np.stack(actions, axis=0)
            rewards = np.stack(rewards, axis=0)
            poststates = np.stack(poststates, axis=0)
            postbirds = np.stack(postbirds, axis=0)
            terminals = np.stack(terminals, axis=0)
            weights = np.stack(weights, axis=0)
            targets = np.stack(targets, axis=0)

            self.pred_net.optimize(prestates, prebirds, actions,
                    targets, self.lr, weights)
            print("Optimized")
            step += 1
            if step % 100 == 100 - 1:
                self.target_net.run_copy()
                print(step)
            if step > self.pretrain_steps and self.eps > self.min_eps and (self.eps - self.step) >= self.min_eps:
                self.eps = self.eps - self.step
                if self.beta < 1:
                    self.beta = self.beta - self.beta_step
                    if self.beta > 1:
                        self.beta = 1

        print("SAVE MODEL")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        print("SAVE DONE")
        return
      except:
        print("SAVE MODEL")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        print("SAVE DONE")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        return

    def train(self):
      try:
        self.restore_model()
        if os.path.exists(os.path.join(self.save_dir, 'replay.p')):
            with open(os.path.join(self.save_dir, 'replay.p'), 'rb') as file:
                try:
                    replay = pickle.load(file)
                    self.replay = replay
                except:
                    print("No replay")
                    return
                    pass
                print(self.replay.size, "CURRENT MEMORY SIZE")
        self.target_net.run_copy()
        step = 0
        for epi in range(self.n_steps):
            terminal = False
            action = -1
            reward = 0
            state = None
            prev_state = state
            birdtype = np.zeros(4)
            prev_birdtype = birdtype
            first_shot = True
            while not terminal:
                step += 1
                print("EPI", epi)
                print("eps", self.eps)
                terminal, reward, state, birdtype_, n_birds = \
                        self.env.get_state_reward()
                birdtype = np.zeros(4)
                if birdtype_ is not None:
                    birdtype[birdtype_] = 1
                    birdtype[3] = n_birds / 10
                if not first_shot:
                    self.replay.add(prev_state, prev_birdtype, action, reward,
                                state, birdtype, terminal)
                if not terminal:
                    prev_state = state
                    prev_birdtype = birdtype
                    action = self.pred_net.calc_eps_greedy_actions(
                            state, birdtype, self.eps)
                    # action idx -> theta, v
                    angle, taptime = self.action_space[int(action)]
                    self.env.act(angle, taptime)
                    first_shot = False

                print("OPTIMIZE")
                for j in range(50):
                    prestates = []
                    prebirds = []
                    actions = []
                    rewards = []
                    poststates = []
                    postbirds = []
                    terminals = []
                    weights = []
                    targets = []

                    for i in range(self.n_batch):
                        prestate, prebird, action_, reward_, poststate, postbird,\
                                terminal_, weight, target = self.replay.sample_one(
                                        self.pred_net,
                                        self.target_net,
                                        self.discount,
                                        self.beta)
                        if reward_ <= 0:
                            reward_ = -1
                        prestates.append(prestate)
                        prebirds.append(prebird)
                        actions.append(action_)
                        rewards.append(reward_)
                        poststates.append(poststate)
                        postbirds.append(postbird)
                        terminals.append(terminal_)
                        weights.append(weight)
                        targets.append(target)

                    prestates = np.stack(prestates, axis=0)
                    prebirds = np.stack(prebirds, axis=0)
                    actions = np.stack(actions, axis=0)
                    rewards = np.stack(rewards, axis=0)
                    poststates = np.stack(poststates, axis=0)
                    postbirds = np.stack(postbirds, axis=0)
                    terminals = np.stack(terminals, axis=0)
                    weights = np.stack(weights, axis=0)
                    targets = np.stack(targets, axis=0)
                    # prestates, prebirds, actions, rewards, poststates, postbirds, terminals = self.replay.sample()

                    #if self.double_q:
                    #    max_actions = self.pred_net.calc_actions(poststates, postbirds)
                    #    targets = self.target_net.calc_outputs_with_idx(
                    #            poststates, postbirds,
                    #        [[idx, pred_a] for idx, pred_a in enumerate(max_actions)])
                    #else:
                    #    targets = self.target_net.calc_max_outputs(poststates, postbirds)
                    #targets = targets * np.where(terminals, 0, 1)
                    #targets = rewards + self.discount * targets

                    self.pred_net.optimize(prestates, prebirds, actions,
                            targets, self.lr, weights)
                print("Optimized")
            if step % self.update_freq == self.update_freq - 1:
                self.target_net.run_copy()
            if (step + 1) % 100 == 0:
                print("SAVE MODEL")
                saver = tf.train.Saver()
                saver.save(self.sess, self.save_path)
                with open(os.path.join(self.save_dir, 'replay.p'), 'wb') as f:
                    pickle.dump(self.replay, f)
                print("SAVE DONE")
            if step > self.pretrain_steps and self.eps > self.min_eps and (self.eps - self.step) >= self.min_eps:
                self.eps = self.eps - self.step
                self.beta = self.beta - self.beta_step
                if self.beta > 1:
                    self.beta = 1

        print("SAVE MODEL")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        with open(os.path.join(self.save_dir, 'replay.p'), 'wb') as f:
            pickle.dump(self.replay, f)
        print("SAVE DONE")
        return
      except:
        print("SAVE MODEL")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)
        with open(os.path.join(self.save_dir, 'replay.p'), 'wb') as f:
            pickle.dump(self.replay, f)
        print("SAVE DONE")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        return

    def competition(self):
        if not self.restore_model():
            return
        lost = False
        while not lost:
            terminal = False
            action = -1
            state = None
            while not terminal:
                terminal, gamestate, state, birdtype_, n_birds = \
                        self.env.get_state()
                birdtype = np.zeros(4)
                if birdtype_ is not None:
                    birdtype[birdtype_] = 1
                    birdtype[3] = n_birds / 10
                if not terminal:
                    action = self.pred_net.calc_eps_greedy_actions(
                            state, birdtype)
                    angle, taptime = self.action_space[int(action)]
                    self.env.act(angle, taptime)
                else:
                    lost = (gamestate == 2)
        print("Game Over")




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
