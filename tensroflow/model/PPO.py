import gymnasium
import soulsgym
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils
import config.constant as constant
import random
import logging

C_LR = 0.002
A_LR = 0.001


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, constant.S_DIM], 'State')
        self.hc = tf.placeholder(tf.float32, [None, constant.S_DIM * 2], "hidden_cell")
        self.legal_action = tf.placeholder(tf.float32, [None, constant.A_NUM], "legal_action")
        self.img = tf.placeholder(tf.float32, [None, 90, 160, 3], "image")
        
        with tf.variable_scope("cnn"):
            # conv 
            w_c1 = tf.get_variable("w_c1", [90, 160, 3, 5])
            w_c2 = tf.get_variable("w_c2", [87, 157, 5, 10])

            conv1 = tf.nn.conv2d(self.img, w_c1, padding="SAME")
            maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides = [1,1,1,1],padding="VALID")
            conv2 = tf.nn.conv2d(maxpool1, w_c2, padding="SAME")
            maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides = [1,1,1,1], padding="VALID")

            self.image_flatten = tf.layers.flatten(inputs = maxpool2)

            self.image_latent = tf.layers.dense(self.image_flatten, 20, tf.nn.relu, kernel_initializer='random_uniform')
        

        self.obs = tf.concat([self.tfs, self.legal_action, self.image_latent], axis=1) 

        # self.wx = tf.placeholder(tf.float32, [constant.S_DIM, constant.H_DIM * 4], "lstm_wx")
        # self.wh = tf.placeholder(tf.float32, [constant.H_DIM, constant.H_DIM * 4], "lstm_wh")
        # self.b = tf.placeholder(tf.float32, [constant.H_DIM *4], "b")

        with tf.variable_scope("input"):
            wx = tf.get_variable("wx", [constant.S_DIM + constant.A_NUM + 20, constant.H_DIM * 4], initializer=tf.random_normal_initializer(mean=0, stddev=1))
            wh = tf.get_variable("wh", [constant.H_DIM, constant.H_DIM * 4], initializer=tf.random_normal_initializer(mean=0, stddev=1))
            b = tf.get_variable("b", [constant.H_DIM *4], initializer=tf.random_normal_initializer(mean=0, stddev=1))

            c, h = tf.split(axis=1, num_or_size_splits=2, value=self.hc)
            x = tf.cast(self.obs, tf.float32)
            c = tf.cast(c, tf.float32)
            h = tf.cast(h, tf.float32)
            z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
            i, f, o, g = tf.split(axis=1, num_or_size_splits=4, value=z)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            g = tf.tanh(g)
            c = f *c + i * g
            h = o * tf.tanh(c)
            x = h
            self.next_hc = tf.concat(axis=1, values=[c, h])
            
            self.latent = h

        # critic
        with tf.variable_scope("critic"):
            l1 = tf.layers.dense(self.latent, 20, tf.nn.relu, kernel_initializer='random_uniform')
            self.v = tf.layers.dense(l1 , 1, kernel_initializer='random_uniform')

            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
        
        # actor
        with tf.variable_scope("pi"):
            l1 = tf.layers.dense(self.latent, 100, tf.nn.tanh, trainable=True, kernel_initializer='random_uniform')
            self.action_dist = tf.layers.dense(l1, 20, tf.nn.softmax, trainable=True, kernel_initializer='random_uniform')
            self.action_dist = self.action_dist * self.legal_action
        self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi")
        
        with tf.variable_scope("old_pi"):
            l1 = tf.layers.dense(self.latent, 100, tf.nn.relu, trainable=False, kernel_initializer='random_uniform')
            self.old_action_dist = tf.layers.dense(l1, 20, tf.nn.softmax, trainable=False, kernel_initializer='random_uniform')
            self.old_action_dist = self.old_action_dist * self.legal_action
        self.oldpi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="old_pi")

        with tf.variable_scope("sample_action"):
            dist = tf.distributions.Categorical(probs = self.action_dist)
            self.sample_op = dist.sample(1)
        
        with tf.variable_scope("best_action"):
            self.best_action = tf.arg_max(self.action_dist, 1)

        with tf.variable_scope("update_old_pi"):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, 1], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = self.action_dist / (self.old_action_dist + 1e-5)
                surr = ratio * self.tfadv
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-0.2, 1.+0.2)*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def choose_action(self, s, hc, la, img):
        np.reshape(hc, [-1, 52])
        legal_action, hc, a = self.sess.run([self.legal_action, self.next_hc, self.sample_op], {self.tfs: s, self.hc : hc, self.legal_action: la, self.img : img})
        a = np.int64(a[0][0])
        
        return np.clip(a, 0, 19), hc, legal_action
    
    def choose_best_action(self, s, hc, la):
        np.reshape(hc, [-1, 52])
        legal_action, hc, a = self.sess.run([self.legal_action, self.next_hc, self.best_action], {self.tfs: s, self.hc : hc, self.legal_action: la})
        
        return np.clip(a, 0, 19), hc, legal_action
        
    def get_v(self, s, hc, la):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.hc : hc, self.legal_action: la})[0, 0]

    def update(self, s, a, r, hc, la, img, ep):
        self.sess.run(self.update_oldpi_op)

        adv = self.sess.run(self.advantage, {self.tfs: s, self.hc: hc, self.tfdc_r: r, self.legal_action : la, self.img : img})

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.hc: hc, self.tfa: a, self.tfadv: adv, self.legal_action: la, self.img : img}) for _ in range(10)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.hc: hc, self.tfdc_r: r, self.legal_action:la, self.img : img}) for _ in range(10)]

        self.save_model(ep)

    def save_model(self, ep):
        self.saver.save(self.sess, "./checkpoint/souls_model")

    def load_model(self, ep):
        self.saver.save(self.sess, tf.train.latest_checkpoint('./checkpoint'))