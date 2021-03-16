from absl import app
from absl import flags
from absl import logging

from datetime import datetime
import gin
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl
import numpy as np
import os

from tunable_agents import utility, agent
import tensorflow as tf
from tf_agents.environments import py_environment


@gin.configurable(allowlist=['collect_episodes'])
def collection_step(env: py_environment.PyEnvironment, tf_agent: agent.DQNAgent,
                    replay_memory: agent.ReplayMemory, reward_tracker: agent.RewardTracker,
                    collect_episodes: int) -> None:
    """Samples transitions with the given Driver."""
    if not collect_episodes:
        return

    for _ in range(collect_episodes):
        # Reset env
        ts = env.reset()
        observations = ts.observation

        episode_reward = 0
        done = False
        while not done:
            action = tf_agent.epsilon_greedy_policy(observations)
            ts = env.step(action)
            next_obs, reward, done = ts.observation, ts.reward, ts.is_last()

            replay_memory.append((observations, action, reward, next_obs, done))
            observations = next_obs

            episode_reward += reward

        reward_tracker.append(episode_reward)

    return episode_reward


@gin.configurable(allowlist=["batch_size", "train_steps"])
def training_step(tf_agent: agent.DQNAgent, replay_memory: agent.ReplayMemory, batch_size: int,
                  train_steps: int) -> None:
    if not train_steps:
        return

    for c in range(train_steps):
        if c % (train_steps / 10) == 0 and c != 0:
            pass
        experiences = replay_memory.sample(batch_size)
        tf_agent.training_step(experiences)


@gin.configurable
def plot_learning_curve(reward_tracker: agent.RewardTracker,
                        average_reward_window: int,
                        image_path=None,
                        csv_path=None):
    """
        Plot the rewards per episode collected during training
        """
    reward_data = reward_tracker.get_reward_data()
    x = reward_data[:, 0]
    y = reward_data[:, 1]

    # Save raw reward data
    if csv_path:
        np.savetxt(csv_path, reward_data, delimiter=",")

    # Compute moving average
    tracker = agent.MovingAverage(maxlen=average_reward_window)
    mean_rewards = np.zeros(len(reward_data))
    for i, (_, reward) in enumerate(reward_data):
        tracker.append(reward)
        mean_rewards[i] = tracker.mean()

    # Create plot
    colour_palette = get_cmap(name='Set1').colors
    plt.figure(figsize=(13, 8), dpi=400)
    plt.plot(x, y, alpha=0.2, c=colour_palette[0])
    plt.plot(x[average_reward_window // 2:], mean_rewards[average_reward_window // 2:], c=colour_palette[0])
    plt.xlabel('Episode')
    plt.ylabel('Reward per Episode')
    plt.grid(True, ls=':')

    # Save plot
    if image_path:
        plt.savefig(image_path, dpi=400)
    plt.close()


@gin.configurable
def train_eval(
    # Params for experiment identification
    root_dir: str,
    training_id: str,
    model_id: str,
    env_id: str,
    # Params for training process
    num_iterations: int,
    target_update_period: int,
    # Params for eval
    eval_interval: int,
    num_eval_episodes: int,
    # Param for checkpoints
    checkpoint_interval: int,
    replay_size: int,
):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    experiment_dir = os.path.join(root_dir, "-".join([model_id, env_id, training_id]))
    model_dir = os.path.join(experiment_dir, 'model')
    plots_dir = os.path.join(experiment_dir, 'plots')
    model_path = os.path.join(model_dir, 'dqn_model.h5')
    image_path = os.path.join(plots_dir, 'reward_plot.png')
    csv_path = os.path.join(plots_dir, 'reward_data.csv')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(plots_dir)

    tf.profiler.experimental.server.start(6008)

    # create the enviroment
    env = utility.create_environment()
    epoch_counter = tf.Variable(0, trainable=False, name='Epoch', dtype=tf.int64)
    # Epsilon implementing decaying behaviour for the two agents
    decaying_epsilon = utility.decaying_epsilon(step=epoch_counter)
    # create an agent and a network
    tf_agent = agent.DQNAgent(epsilon=decaying_epsilon, obs_spec=env.observation_spec())
    # replay buffer
    replay_memory = agent.ReplayMemory(maxlen=replay_size)
    reward_tracker = agent.RewardTracker()

    start_time = datetime.now()

    # Initial collection
    with gin.config_scope('initial_step'):
        collection_step(env=env,
                        tf_agent=tf_agent,
                        replay_memory=replay_memory,
                        reward_tracker=reward_tracker)

    for _ in range(num_iterations):
        epoch_counter.assign_add(1)
        tf.summary.scalar(name='Epsilon', data=decaying_epsilon(), step=epoch_counter)

        episode_reward = collection_step(env=env,
                                         tf_agent=tf_agent,
                                         replay_memory=replay_memory,
                                         reward_tracker=reward_tracker)

        training_step(tf_agent=tf_agent, replay_memory=replay_memory)

        avg_reward = reward_tracker.mean()
        tf.summary.scalar(name='Average Reward', data=avg_reward, step=epoch_counter)

        print("\rTime: {}, Episode: {}, Reward: {}, Avg Reward {}, eps: {:.3f}".format(
            datetime.now() - start_time, epoch_counter.numpy(), episode_reward, avg_reward,
            decaying_epsilon().numpy()),
              end="")

        # Copy weights from main model to target model
        if epoch_counter.numpy() % target_update_period == 0:
            tf_agent.update_target_model()

        # Checkpointing
        if epoch_counter.numpy() % checkpoint_interval == 0:
            tf_agent.save_model(model_path)
            plot_learning_curve(reward_tracker=reward_tracker, image_path=image_path, csv_path=csv_path)

        # Evaluation Run
        if epoch_counter.numpy() % eval_interval == 0:
            pass


def main(_):
    logging.set_verbosity(logging.INFO)
    utility.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    train_eval(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing logs/summaries/checkpoints.')
    flags.DEFINE_multi_string(
        'gin_files', [], 'List of paths to gin configuration files (e.g.'
        '"configs/hanabi_rainbow.gin").')
    flags.DEFINE_multi_string(
        'gin_bindings', [], 'Gin bindings to override the values set in the config files '
        '(e.g. "train_eval.num_iterations=100").')
    FLAGS = flags.FLAGS

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    mpl.rcParams[
        'agg.path.chunksize'] = 1_000  # Needed to avoid a matplotlib backend error when plotting high dpi images
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)

    flags.mark_flag_as_required('root_dir')
    app.run(main)
