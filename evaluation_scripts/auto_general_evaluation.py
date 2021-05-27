from absl import app
from absl import flags
from absl import logging

import copy
import gin
import itertools
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from tf_agents.environments import py_environment
from tunable_agents import utility, agent

ENV_KWARGS = {
    'linear_env': {
        "utility_type": "linear"
    },
    'cum_linear_env': {
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
    },
    'linear_threshold_env': {
        "utility_type": 'linear_threshold'
    },
    'cum_linear_threshold_env': {
        "utility_type": 'linear_threshold',
        "cumulative_rewards_flag": True
    },
    'dual_threshold_env': {
        "utility_type": 'dual_threshold'
    },
    'cum_dual_threshold_env': {
        "utility_type": 'dual_threshold',
        "cumulative_rewards_flag": True
    },
    'linear_dual_threshold_env': {
        "utility_type": 'linear_dual_threshold'
    },
    'cum_linear_dual_threshold_env': {
        "utility_type": 'linear_dual_threshold',
        "cumulative_rewards_flag": True
    }
}

ENVS = [
    "cum_linear_threshold_env", "cum_linear_dual_threshold_env", "cum_linear_env", "cum_target_env",
    "cum_dual_threshold_env", "cum_threshold_env", "linear_dual_threshold_env", "linear_threshold_env", "linear_env",
    "target_env", "dual_threshold_env", "threshold_env"
]
MODELS = ["64_64_model", "128_128_64_model", "256_128_128_64_64_model", "512_256_256_128_128_64_model"]
TRAINING_IDS = ["replication" + train_id for train_id in ["", "-1", "-2", "-3", "-4", "-5"]]

SAMPLINGS = ["", "dense", "continuous"]
UTILITY_EPISODES = 100_000
REWARD_VECTOR_EPISODES = 400_000
LOCKS_PATH = Path("locks")


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    pass


def generate_results_path(results_dir: Path, env: str, model: str, training_id: str, lin_thresh: bool,
                          reward_vector: bool, sampling: str) -> Path:
    experiment_id = "-".join([model, env, training_id])
    results_file_name = experiment_id + ".npy"
    if sampling:
        results_file_name = results_file_name.split("-")
        results_file_name[1] = results_file_name[1][:-4] + "_" + sampling + "_env"
        results_file_name = "-".join(results_file_name)
    if lin_thresh:
        results_file_name = results_file_name.split("-")
        results_file_name[1] = results_file_name[1][:-4] + "_linear_env"
        results_file_name = "-".join(results_file_name)

    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    aggregation_dir = results_dir.joinpath("reward_vector" if reward_vector else "utility_results",
                                           "-".join(results_file_name.split("-")[:2]))
    results_path = aggregation_dir.joinpath(results_file_name)

    return results_path


def check_lock(results_path: Path, reward_vector: bool) -> bool:
    """
    Checks if the evaluation run defined by the results_path passed needs to be executed, and if so
    checks that there isn't a lock already on it.

    Returns:
        bool: Whether to execute the evaluation run or not.
    """
    if results_path.exists():
        sampled_size = np.load(results_path, allow_pickle=True).shape[0]
        if sampled_size >= (REWARD_VECTOR_EPISODES if reward_vector else UTILITY_EPISODES):
            return False

    lock_name = results_path.name[:-4] + ("-reward_vector" * reward_vector) + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    if lock_path.exists():
        return False

    return True


def acquire_lock(results_path: Path, reward_vector: bool) -> None:
    """
    Adds a lock on the current evaluation run.
    """
    if not LOCKS_PATH.exists():
        LOCKS_PATH.mkdir()
    lock_name = results_path.name[:-4] + ("-reward_vector" * reward_vector) + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    np.save(lock_path, None)


def release_lock(results_path: Path, reward_vector: bool) -> None:
    """
    Releases the lock on the current evaluation run.
    """
    lock_name = results_path.name[:-4] + ("-reward_vector" * reward_vector) + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    os.remove(lock_path)


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

        assert np.isclose(episode_reward, env._prev_step_utility, atol=1e-05)
        if reward_vector:
            results.append([observations['utility_representation'], np.copy(env._cumulative_rewards)])
        else:
            results.append(episode_reward)

    if reward_vector:
        results = np.array(results, dtype='object')
    else:
        results = np.array(results)

    return results


def main(_):
    logging.set_verbosity(logging.INFO)

    for env_type, model, training_id, lin_thresh, reward_vector, sampling in itertools.product(
            ENVS, MODELS, TRAINING_IDS, [True, False], [True, False], SAMPLINGS):
        if lin_thresh and not "threshold" in env_type:
            continue  # Can't evaluate a non threshold agent on threshold utilities.
        if sampling and "target" in env_type:
            continue  # Can't select sampling for target utility function.
        if (not lin_thresh) and ("linear_threshold" in env_type or "linear_dual_threshold" in env_type):
            continue  # Can't sample out of distribution for a linear (dual) threshold agent.

        experiment_dir = os.path.join(FLAGS.root_dir, "-".join([model, env_type, training_id]))
        model_path = os.path.join(experiment_dir, 'model', 'dqn_model.h5')
        results_path = generate_results_path(Path(FLAGS.results_dir),
                                             env=env_type,
                                             model=model,
                                             training_id=training_id,
                                             lin_thresh=lin_thresh,
                                             reward_vector=reward_vector,
                                             sampling=sampling)

        if not check_lock(results_path=results_path, reward_vector=reward_vector):
            continue

        acquire_lock(results_path=results_path, reward_vector=reward_vector)

        print(f"\n\n{results_path}\n")

        # Loading appropriate gin configs for the environment and this experiment
        gin.clear_config()
        qnet_gin, env_gin = experiment_dir.split("/")[-1].split("-")[:2]
        gin_config_path = Path("tunable-agents-MORL/configs/")
        gin_files = [
            gin_config_path.joinpath("qnets/", qnet_gin + ".gin"),
            gin_config_path.joinpath("envs/", "linear_env.gin")
        ]
        utility.load_gin_configs(gin_files, [])

        # Loading trained agent model
        env_kwargs = copy.copy(ENV_KWARGS[env_gin])
        if lin_thresh and (not "linear" in env_kwargs["utility_type"]):
            env_kwargs["utility_type"] = "linear_" + env_kwargs["utility_type"]
        elif (not lin_thresh) and ("linear" in env_kwargs["utility_type"]) and ("threshold" in env_type):
            env_kwargs["utility_type"] = env_kwargs["utility_type"][7:]
        env_kwargs["sampling"] = sampling
        env = utility.create_environment(**env_kwargs)
        tf_agent = agent.DQNAgent(epsilon=0, obs_spec=env.observation_spec())
        tf_agent.load_model(model_path)

        # Evaluating the agent
        n_episodes = REWARD_VECTOR_EPISODES if reward_vector else UTILITY_EPISODES
        if results_path.exists():
            n_episodes -= np.load(results_path, allow_pickle=True).shape[0]
        results = eval_agent(env, tf_agent, n_episodes, reward_vector)

        # Save results
        if results_path.exists():
            np.save(results_path, np.concatenate((np.load(results_path, allow_pickle=True), results), axis=0))
        else:
            if not results_path.parent.exists():
                results_path.parent.mkdir(parents=True)
            np.save(results_path, results)

        release_lock(results_path=results_path, reward_vector=reward_vector)


if __name__ == '__main__':
    flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory where all the trained agents are saved')
    flags.DEFINE_string('results_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                        'Root directory for writing the results')
    FLAGS = flags.FLAGS
    flags.mark_flag_as_required('root_dir')
    flags.mark_flag_as_required('results_dir')

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    app.run(main)
