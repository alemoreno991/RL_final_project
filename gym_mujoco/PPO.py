import numpy as np
import tensorflow as tf


class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs0, act, rwd):
        self.ep_obs.append(obs0)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def covert_to_array(self):
        array_obs = np.vstack(self.ep_obs)
        array_act = np.vstack(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, act_dim, name, trainable=True):
        self.act_dim = act_dim
        self.name = name
        self.trainable = trainable

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h0 = tf.layers.dense(obs, 128, activation=tf.nn.relu, trainable=self.trainable)
            h1 = tf.layers.dense(h0, 32, activation=tf.nn.relu, trainable=self.trainable)
            mu = 2 * tf.layers.dense(h1, self.act_dim, activation=tf.nn.tanh, trainable=self.trainable)
            sigma = tf.layers.dense(h1, self.act_dim, activation=tf.nn.softplus, trainable=self.trainable)
            pi = tf.distributions.Normal(loc=mu, scale=sigma)
            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            return pi, param

    def pi(self, obs, reuse=False):
        pi = self.step(obs, reuse)
        return pi


class ValueNetwork(object):
    def __init__(self, name):
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h0 = tf.layers.dense(inputs=obs, units=128, activation=tf.nn.relu)
            h1 = tf.layers.dense(inputs=h0, units=32, activation=tf.nn.relu)
            value = tf.layers.dense(inputs=h1, units=1, activation=None)
            return value

    def get_value(self, obs, reuse=False):
        value = self.step(obs, reuse)
        return value


class PPO(object):
    def __init__(self, act_dim, obs_dim, lr_actor, lr_value, gamma, clip_range):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.gamma = gamma
        self.clip_range = clip_range

        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name="observation")
        self.ACT = tf.placeholder(tf.float32, [None, self.act_dim], name="action")
        self.Q_VAL = tf.placeholder(tf.float32, [None, 1], name="q_value")
        self.ADV = tf.placeholder(tf.float32, [None, 1], name="advantage")

        self.memory = Memory()

        old_actor = ActorNetwork(self.act_dim, 'old_actor', trainable=False)
        old_pi, old_pi_param = old_actor.pi(self.OBS) 

        actor = ActorNetwork(self.act_dim, 'actor')
        pi, pi_param = actor.pi(self.OBS)

        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]
        self.action = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), 0, 1)[0]
        with tf.variable_scope('actor_loss'):
            ratio = pi.prob(self.ACT) / (old_pi.prob(self.ACT) + 1e-4)
            pg_losses= self.ADV * ratio
            pg_losses2 = self.ADV * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss)

        value = ValueNetwork('critic')
        self.value = value.get_value(self.OBS)
        with tf.variable_scope('critic_loss'):
            self.advantage = self.Q_VAL - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))         
        self.value_train_op = tf.train.AdamOptimizer(self.lr_value).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis, :]
        action = self.sess.run(self.action, feed_dict={self.OBS: obs})

        value = self.sess.run(self.value, feed_dict={self.OBS: obs})[0, 0]
        return action, value

    def learn(self, last_value, done):
        self.sess.run(self.syn_old_pi)

        obs, act, rwd = self.memory.covert_to_array()
        q_value = self.compute_q_value(last_value, done, rwd)

        adv = self.sess.run(self.advantage, {self.OBS: obs, self.Q_VAL: q_value})

        [self.sess.run(self.value_train_op,
                       {self.OBS: obs, self.Q_VAL: q_value}) for _ in range(32)]
        [self.sess.run(self.actor_train_op,
                          {self.OBS: obs, self.ACT: act, self.ADV: adv}) for _ in range(32)]

        self.memory.reset()

    def compute_q_value(self, last_value, done, rwd):
        q_value = np.zeros_like(rwd)
        value = 0 if done else last_value
        for t in reversed(range(0, len(rwd))):
            value = value * self.gamma + rwd[t]
            q_value[t] = value
        return q_value[:, np.newaxis]
