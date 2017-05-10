import os

import gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dqn.agent import Agent
from utils.utils import atari_preprocessing, get_histo

flags = tf.app.flags

# Model
flags.DEFINE_integer("history_length", 4, "History length to see")
flags.DEFINE_integer("memory_size", 1000000, "")
flags.DEFINE_float("gamma", 0.99, "")

# Training
flags.DEFINE_integer("num_steps", 10000000, "Number of steps to run/train")
flags.DEFINE_integer("target_network_update_step", 10000, "Period to update target q network")
flags.DEFINE_integer("learn_start", 50000, "Steps to start learning")

flags.DEFINE_integer("batch_size", 32, "")

flags.DEFINE_integer("min_delta", -1, "")
flags.DEFINE_integer("max_delta", 1, "")

flags.DEFINE_float("initial_learning_rate", 0.00025, "")
flags.DEFINE_float("learning_rate_minimum", 0.00025, "")
flags.DEFINE_float("learning_rate_decay", 0.96, "")
flags.DEFINE_float("learning_rate_decay_step", 50000, "")

flags.DEFINE_float("epsilon_start", 1.0, "")
flags.DEFINE_float("epsilon_end", 0.1, "")
flags.DEFINE_integer("epsilon_end_step", 50000, "")

flags.DEFINE_integer("update_frequency", 4, "")

# Environment
flags.DEFINE_string("env_name", "Breakout-v0", "Gym environment name to use")
flags.DEFINE_integer("screen_width", 84, "")
flags.DEFINE_integer("screen_height", 84, "")
flags.DEFINE_integer("action_repeat", 4, "")

# Etc
flags.DEFINE_boolean("is_train", True, "Whether to do training or testing")
flags.DEFINE_integer("random_seed", 12345, "Random seed")
flags.DEFINE_boolean("display", False, "Whether to display the game screen or not")
flags.DEFINE_string("train_dir", "checkpoints/breakout", "")
flags.DEFINE_string("gpu_id", "0", "")

FLAGS = flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)

TEST_STEP = 50000
TENSORBOARD_STEP = 50
SAVE_STEP = 500000


def _init_log_vars():
    return {"num_games": 0,
            "ep_reward": 0.,
            "ep_rewards": [],
            "actions": []}


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

    train_dir = FLAGS.train_dir
    if not tf.gfile.Exists(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options
    )

    with tf.Session(config=session_config) as sess:
        env = gym.make(FLAGS.env_name)
        agent = Agent(FLAGS, env.action_space.n)

        # Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Setup logger/saver
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # Variables for logging
        log_vars = _init_log_vars()

        done = True
        for step in tqdm(range(FLAGS.num_steps), ncols=70):
            if done:
                env.reset()
                log_vars["num_games"] += 1
                log_vars["ep_rewards"].append(log_vars["ep_reward"])
                log_vars["ep_reward"] = 0.
                action = 0

            reward = 0.
            for _ in xrange(FLAGS.action_repeat):
                raw_observation, one_reward, done, info = env.step(action)
                reward += one_reward
                if done:
                    reward -= 1.
                    break
            observation = atari_preprocessing(raw_observation, FLAGS.screen_width, FLAGS.screen_height)
            log_vars["ep_reward"] += reward

            if FLAGS.display:
                env.render()

            if FLAGS.is_train:
                action, epsilon, summary_str = agent.train(observation, reward, done, step, sess)
            else:
                action, epsilon, summary_str = agent.predict(observation, sess)
            log_vars["actions"].append(action)

            # Log performance periodically
            if (step + 1) % TEST_STEP == 0:
                ep_rewards = log_vars["ep_rewards"]
                num_games = log_vars["num_games"]
                actions = log_vars["actions"]
                try:
                    max_r = np.max(ep_rewards)
                    min_r = np.min(ep_rewards)
                    avg_r = np.mean(ep_rewards)
                except:
                    max_r = 0.0
                    min_r = 0.0
                    avg_r = 0.0

                format_str = "[Step %d] avg_r: %.4f, max_r: %.4f, min_r: %.4f, # games: %d, epsilon: %.4f" % \
                    (step, avg_r, max_r, min_r, num_games, epsilon)
                tf.logging.info(format_str)

                summary = tf.Summary()
                summary.value.add(tag="avg_r", simple_value=avg_r)
                summary.value.add(tag="max_r", simple_value=max_r)
                summary.value.add(tag="min_r", simple_value=min_r)
                summary.value.add(tag="num_games", simple_value=num_games)
                summary.value.add(tag="epsilon", simple_value=epsilon)
                summary.value.add(tag="actions", histo=get_histo(actions))
                summary_writer.add_summary(summary, step)

                log_vars = _init_log_vars()

            # Update Tensorboard
            if (step + 1) % TENSORBOARD_STEP == 0 and step > FLAGS.learn_start and summary_str:
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically
            if (step + 1) % SAVE_STEP == 0:
                tf.logging.info("Save checkpoint at %d step" % step)
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=agent.global_step.eval())

        env.close()


if __name__ == "__main__":
    tf.app.run()
