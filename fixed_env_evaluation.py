from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

from tunable_agents import utility, agent
from tunable_agents.environments.gathering_env.gathering_env import GatheringWrapper
from tunable_agents.environments.utility_functions import LinearUtility, ThresholdUtility

ENV_KWARGS = {
    'replication_env': {
    },  # fixed_env config is based off of replication env so no change to kwargs is needed
    'cum_rewards_env': {
        "cumulative_rewards_flag": True
    },
    'threshold_env': {
        "utility_type": 'threshold'
    },
    'cum_threshold_env': {
        "utility_type": 'threshold',
        "cumulative_rewards_flag": True
    }
}

LINEAR_UTILITIES = [
    LinearUtility(weights=np.array([-1, -5, r0, r1, r2, r3], dtype=np.float32))
    for r0 in np.arange(-20, 21, step=5)
    for r1 in np.arange(-20, 21, step=5)
    for r2 in np.arange(-20, 21, step=5)
    for r3 in np.arange(-20, 21, step=5)
    if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
]

THRESHOLD_UTILITIES = [
    ThresholdUtility(thresholds_and_ceofficients=np.array(
        [[0, 0, thresh0, thresh1, thresh2, thresh3], util.utility_repr], dtype=np.float32))
    for thresh0 in np.arange(0, 4)
    for thresh1 in np.arange(0, 4)
    for thresh2 in np.arange(0, 4)
    for thresh3 in np.arange(0, 4)
    for util in LINEAR_UTILITIES
]


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    pass


def fixed_env_eval(env_kwargs: dict, tf_agent: agent.DQNAgent) -> np.ndarray:
    utilities = LINEAR_UTILITIES if "linear" in env_kwargs.values() else THRESHOLD_UTILITIES
    results = []
    for utility in tqdm(utilities):
        env = GatheringWrapper(utility_repr=utility.utility_repr, **env_kwargs)
        ts = env.reset()
        observations = ts.observation

        done = False
        while not done:
            action = tf_agent.greedy_policy(observations)
            ts = env.step(action)
            observations, done = ts.observation, ts.is_last()

        results.append(env._cumulative_rewards)

    return results


def main(_):
    logging.set_verbosity(logging.INFO)
    gin_files = ["tunable-agents-MORL/configs/envs/fixed_env.gin"]
    if "64_64_model" in FLAGS.experiment_dir:
        gin_files.append("tunable-agents-MORL/configs/qnets/replication_model.gin")
    elif "128_128_64_model" in FLAGS.experiment_dir:
        gin_files.append("tunable-agents-MORL/configs/qnets/128_128_64_model.gin")
    utility.load_gin_configs(gin_files, [])

    experiment_dir = FLAGS.experiment_dir
    model_dir = os.path.join(experiment_dir, 'model')
    model_path = os.path.join(model_dir, 'dqn_model.h5')

    for key in ENV_KWARGS:
        if key in experiment_dir:
            env_kwargs = ENV_KWARGS[key]
    
    env = GatheringWrapper(**env_kwargs)
    tf_agent = agent.DQNAgent(epsilon=0, obs_spec=env.observation_spec())

    tf_agent.load_model(model_path)

    results = fixed_env_eval(env_kwargs, tf_agent)

    results_dir = os.path.join(FLAGS.results_dir, "fixed_env_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_path = os.path.join(results_dir, experiment_dir.split("/")[-1] + ".npy")
    np.save(results_path, results)


if __name__ == '__main__':
    flags.DEFINE_string('experiment_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Directory where the agent is saved.')
    flags.DEFINE_string('results_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing the results')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('experiment_dir')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app.run(main)
