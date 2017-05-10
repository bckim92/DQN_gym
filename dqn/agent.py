import random

import tensorflow as tf

from dqn.replay_memory import ReplayMemory, History

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


def clipped_error(x):
    """Huber loss"""
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class Agent(object):
    def __init__(self, config, action_space):
        self.replay_memory = ReplayMemory(config)
        self.history = History(config)
        self.config = config
        self.action_space = action_space
        self.train_counter = 0

        #self.w_initializer = tf.truncated_normal_initializer(0, 0.02)
        #self.w_initializer = tf.uniform_unit_scaling_initializer(1.0)
        self.w_initializer = tf.contrib.layers.xavier_initializer()
        self.b_initializer = tf.constant_initializer(0.0)

        # Build placeholders
        with tf.name_scope("placeholders"):
            self.current_observation = tf.placeholder(
                tf.float32,
                [None, self.config.screen_height, self.config.screen_width, self.config.history_length]
            )
            self.next_observation = tf.placeholder(
                tf.float32,
                [None, self.config.screen_height, self.config.screen_width, self.config.history_length]
            )
            self.current_action = tf.placeholder(tf.int32, [None])
            self.current_reward = tf.placeholder(tf.float32, [None])
            self.done = tf.placeholder(tf.float32, [None])

        # Build ops
        self.train_op, self.predicted_action, self.target_update_op = self._build(
            self.current_observation, self.next_observation, self.current_action,
            self.current_reward, self.done
        )
        self.summary_op = tf.summary.merge_all()

    def train(self, observation, reward, done, current_step, sess):
        # Update history
        self.history.add(observation)

        # Predict action via epsilon-greedy policy
        epsilon = (self.config.epsilon_end +
                   max(0.0, ((self.config.epsilon_start - self.config.epsilon_end) *
                             (self.config.epsilon_end_step - max(0., current_step - self.config.learn_start)) /
                             self.config.epsilon_end_step)))

        if random.random() < epsilon:
            action = random.randrange(self.action_space)
        else:
            action = sess.run(
                self.predicted_action,
                {self.current_observation: [self.history.get()]}
            )
            action = action[0]

        # Reset history
        if done:
            self.history.reset()

        # Update memory and sample
        self.replay_memory.add(observation, reward, action, done)

        # Update source network
        if current_step > self.config.learn_start:
            if self.train_counter == self.config.update_frequency:
                current_observation, current_action, current_reward, next_observation, current_done = self.replay_memory.sample()
                _, summary_str = sess.run([self.train_op, self.summary_op],
                                          {self.current_observation: current_observation,
                                           self.next_observation: next_observation,
                                           self.current_action: current_action,
                                           self.current_reward: current_reward,
                                           self.done: current_done})
                self.train_counter = 0
            else:
                self.train_counter += 1
                summary_str = None
        else:
            summary_str = None

        # Update target network
        if (current_step + 1) % self.config.target_network_update_step == 0:
            tf.logging.info("Update target network")
            sess.run([self.target_update_op])

        return action, epsilon, summary_str

    def predict(self, observation, sess):
        pass

    def _build(self, current_observation, next_observation, current_action, current_reward, done):
        # Global variables
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)

        # Build network
        source_q = self._build_network(current_observation, 'source', True)
        target_q = self._build_network(next_observation, 'target', False)

        # Compute loss
        action_one_hot = tf.one_hot(current_action, self.action_space, 1.0, 0.0, name="action_one_hot")
        q_acted = tf.reduce_sum(source_q * action_one_hot, reduction_indices=1, name="q_acted")
        max_target_q = tf.reduce_max(target_q, axis=1)
        delta = (1 - done) * self.config.gamma * max_target_q + current_reward - q_acted
        loss = tf.reduce_mean(clipped_error(delta))

        # Optimize
        learning_rate_op = tf.maximum(
            self.config.learning_rate_minimum,
            tf.train.exponential_decay(
                self.config.initial_learning_rate,
                self.global_step,
                self.config.learning_rate_decay_step,
                self.config.learning_rate_decay,
                staircase=True
            )
        )
        train_op = tf.train.RMSPropOptimizer(learning_rate_op, momentum=0.95, epsilon=0.01).minimize(loss, global_step=self.global_step)

        # Update target network
        target_update_op = []
        source_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source')
        target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        for source_variable, target_variable in zip(source_variables, target_variables):
            target_update_op.append(target_variable.assign(source_variable.value()))
        target_update_op = tf.group(*target_update_op)

        # Logging
        predicted_action = tf.argmax(source_q, dimension=1)
        avg_q = tf.reduce_mean(source_q, 0)
        for idx in xrange(self.action_space):
            tf.summary.histogram('q/%s' % idx, avg_q[idx])
        tf.summary.scalar('learning_rate', learning_rate_op)
        tf.summary.scalar('loss', loss)

        return train_op, predicted_action, target_update_op

    def _build_network(self, observation, name='source', trainable=True):
        with tf.variable_scope(name):
            with arg_scope([layers.conv2d, layers.fully_connected],
                           activation_fn=tf.nn.relu,
                           weights_initializer=self.w_initializer,
                           biases_initializer=self.b_initializer,
                           trainable=trainable):
                with arg_scope([layers.conv2d], padding='VALID'):
                    conv1 = layers.conv2d(observation, num_outputs=32, kernel_size=8, stride=4, scope='conv1')
                    conv2 = layers.conv2d(conv1, num_outputs=64, kernel_size=4, stride=2, scope='conv2')
                    conv3 = layers.conv2d(conv2, num_outputs=64, kernel_size=3, stride=1, scope='conv3')

                conv3_shape = conv3.get_shape().as_list()
                conv3_flat = tf.reshape(conv3, [-1, reduce(lambda x, y: x * y, conv3_shape[1:])])

                fc4 = layers.fully_connected(conv3_flat, 512, scope='fc4')
                q = layers.fully_connected(fc4, self.action_space, scope='q')

        return q
