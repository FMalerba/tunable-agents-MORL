from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import os
from tqdm import tqdm

from tf_agents.environments import py_environment
from tunable_agents import utility, agent


ENV_DICT = {
    "cum_rewards_env": "cumulative_rewards_env.gin",
    "cum_target_env": "cumulative_target_env.gin",
    "cum_threshold_env": "cumulative_threshold_env.gin",
    "replication_env": "replication_env.gin",
    "target_env": "target_env.gin",
    "threshold_env": "threshold_utility_env.gin"
}


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    pass


def eval_agent(env: py_environment.PyEnvironment,
               tf_agent: agent.DQNAgent,
               n_episodes: int,
               reward_vector: bool = False) -> np.ndarray:

    results = []
    for _ in tqdm(range(n_episodes)):
        ts = env.reset()
        observations = ts.observation

        episode_reward = 0
        done = False
        while not done:
            action = tf_agent.greedy_policy(observations)
            ts = env.step(action)
            observations, reward, done = ts.observation, ts.reward, ts.is_last()

            episode_reward += reward

        assert np.isclose(episode_reward, env._prev_step_utility, atol=1e-06)
        if reward_vector:
            results.append([observations['utility_representation'], env._cumulative_rewards])
        else:
            results.append(episode_reward)

    if reward_vector:
        results = np.array(results, dtype='object')
    else:
        results = np.array(results)

    return results


def main(_):
    logging.set_verbosity(logging.INFO)

    if FLAGS.linear_threshold and not "threshold" in FLAGS.experiment_dir:
        raise ValueError("Can't evaluate a non threshold agent on threshold utilities.")
    if FLAGS.dense and "target" in FLAGS.experiment_dir:
        raise ValueError("Can't sample densely for target utility function.")

    experiment_dir: str = FLAGS.experiment_dir
    model_dir = os.path.join(experiment_dir, 'model')
    model_path = os.path.join(model_dir, 'dqn_model.h5')

    # Loading appropriate gin configs for the environment and this experiment
    qnet_gin, env_gin = experiment_dir.split("/")[-1].split("-")[:2]
    gin_config_path = "tunable-agents-MORL/configs/"
    gin_files = [
        gin_config_path + "qnets/" + qnet_gin + ".gin", gin_config_path + "envs/" + ENV_DICT[env_gin]
    ]
    gin_bindings = ["GatheringWrapper.utility_type='linear_threshold'"] if FLAGS.linear_threshold else []
    utility.load_gin_configs(gin_files, gin_bindings)
    utility_type = ("dense_" if FLAGS.dense else "") + gin.query_parameter("GatheringWrapper.utility_type")

    # Loading trained agent model
    env = utility.create_environment(utility_type=utility_type)
    tf_agent = agent.DQNAgent(epsilon=0, obs_spec=env.observation_spec())
    tf_agent.load_model(model_path)

    # Evaluating the agent
    results = eval_agent(env, tf_agent, FLAGS.n_episodes, FLAGS.reward_vector)

    # Save results
    results_dir = os.path.join(FLAGS.results_dir, "reward_vector" if FLAGS.reward_vector else "utility_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file_name = experiment_dir.split("/")[-1] + ".npy"
    if FLAGS.linear_threshold:
        results_file_name = results_file_name.split("-")
        results_file_name[1] += "_linear"
        results_file_name = "-".join(results_file_name)
    if FLAGS.dense:
        results_file_name = results_file_name.split("-")
        results_file_name[1] = "dense_" + results_file_name[1]
        results_file_name = "-".join(results_file_name)

    results_path = os.path.join(results_dir, results_file_name)
    if os.path.exists(results_path):
        results = np.append(np.load(results_path, allow_pickle=True), results, axis=0)
    np.save(results_path, results)


if __name__ == '__main__':
    flags.DEFINE_string('experiment_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Directory where the agent is saved.')
    flags.DEFINE_string('results_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing the results')
    flags.DEFINE_integer('n_episodes', 10_000, "Number of episodes to evaluate the agent on")
    flags.DEFINE_bool(
        'reward_vector', False, "Whether to compute results for the reward vector"
        " instead of the utility scalarized result.")
    flags.DEFINE_bool(
        'linear_threshold', False, "Whether to run a ThresholdUtility-type agent on "
        "utilities that are mathematically equivalent to linear ones.")
    flags.DEFINE_bool(
        'dense', False, "Whether to sample utilities using the dense sampling functions.")
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('experiment_dir')
    flags.mark_flag_as_required('results_dir')

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app.run(main)
