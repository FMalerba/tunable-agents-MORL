from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import os
from tqdm import tqdm

from tunable_agents import utility, agent
from tunable_agents.environments.gathering_env.gathering_env import GatheringWrapper
from tunable_agents.environments.utility_functions import LinearUtility, TargetUtility, ThresholdUtility, UtilityFunction

from typing import List

ENV_KWARGS = {
    'replication_env': {
        "utility_type": "linear"
    },
    'cum_rewards_env': {
        "utility_type": "linear",
        "cumulative_rewards_flag": True
    },
    'target_env': {
        "utility_type": 'target'
    },
    'cum_target_env': {
        "utility_type": 'target',
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


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    pass


def utility_list(utility_type: str):
    if utility_type == "linear":
        return [
            LinearUtility(weights=np.array([-1, -5, r0, r1, r2, r3], dtype=np.float32))
            for r0 in np.arange(-20, 21, step=2)
            for r1 in np.arange(-20, 21, step=2)
            for r2 in np.arange(-20, 21, step=2)
            for r3 in np.arange(-20, 21, step=2)
            if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
        ]
    elif utility_type == "threshold":
        return [
            ThresholdUtility(thresholds_and_ceofficients=np.array(
                [[0, 0, thresh0, thresh1, thresh2, thresh3], [-1, -5, r0, r1, r2, r3]], dtype=np.float32))
            for thresh0 in range(3)
            for thresh1 in range(3)
            for thresh2 in range(3)
            for thresh3 in range(3)
            for r0 in np.arange(-20, 21, step=6)
            for r1 in np.arange(-20, 21, step=6)
            for r2 in np.arange(-20, 21, step=6)
            for r3 in np.arange(-20, 21, step=6)
            if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
        ]
    elif utility_type == "target":
        return [
            TargetUtility(target=np.array([31, 31, target0, target1, target2, target3], dtype=np.float32))
            for target0 in range(4)
            for target1 in range(4)
            for target2 in range(3)
            for target3 in range(4)
            if (target0 > 0) or (target1 > 0) or (target2 > 0) or (target3 > 0)
        ]
    raise ValueError("Got unexpected utility_type argument.")


def fixed_env_eval(tf_agent: agent.DQNAgent, utilities: List[UtilityFunction], **env_kwargs) -> np.ndarray:
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

    experiment_dir: str = FLAGS.experiment_dir
    model_dir = os.path.join(experiment_dir, 'model')
    model_path = os.path.join(model_dir, 'dqn_model.h5')

    # Loading appropriate gin configs for the environment and this experiment
    qnet_gin, env_gin = experiment_dir.split("/")[-1].split("-")[:2]
    gin_files = [
        "tunable-agents-MORL/configs/envs/fixed_env.gin",
        "tunable-agents-MORL/configs/qnets/" + qnet_gin + ".gin"
    ]
    utility.load_gin_configs(gin_files, [])

    # Loading trained agent model
    env_kwargs = ENV_KWARGS[env_gin]
    env = GatheringWrapper(**env_kwargs)
    tf_agent = agent.DQNAgent(epsilon=0, obs_spec=env.observation_spec())
    tf_agent.load_model(model_path)

    # Selecting the utilities to run on
    utilities = utility_list(env_kwargs["utility_type"])
    results_dir = os.path.join(FLAGS.results_dir, "fixed_env_results", experiment_dir.split("/")[-1])
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        results_filepath = os.path.join(results_dir, '0.npy')
        run_id = 0
    else:
        run_id = max([int(file[:-4]) for file in os.listdir(results_dir)]) + 1
        results_filepath = os.path.join(results_dir, str(run_id) + ".npy")
    np.save(results_filepath, None)  # To mark that this run is in execution to other processes
    run_utilities = np.array_split(utilities, 10)[run_id].tolist()

    # Evaluating the agent on the fixed environment
    results = fixed_env_eval(tf_agent, run_utilities, **env_kwargs)

    # Save results
    np.save(results_filepath, results)


if __name__ == '__main__':
    flags.DEFINE_string('experiment_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Directory where the agent is saved.')
    flags.DEFINE_string('results_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing the results')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('experiment_dir')
    flags.mark_flag_as_required('results_dir')

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app.run(main)
