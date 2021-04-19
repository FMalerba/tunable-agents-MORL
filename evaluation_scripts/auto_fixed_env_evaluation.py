from absl import app
from absl import flags
from absl import logging

import gin
import itertools
import numpy as np
import os
from pathlib import Path
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
ENVS = [
    "cum_rewards_env", "cum_target_env", "cum_threshold_env", "replication_env", "target_env", "threshold_env"
]
MODELS = ["64_64_model", "128_128_64_model", "256_128_128_64_64_model", "512_256_256_128_128_64_model"]
TRAINING_IDS = ["replication" + train_id for train_id in ["", "-1", "-2", "-3", "-4", "-5"]]

LOCKS_PATH = Path("locks")


@gin.configurable
def train_eval(training_id: str, model_id: str, env_id: str):
    pass


def generate_results_path(results_dir: Path, env: str, model: str, training_id: str, lin_thresh: bool) -> Path:
    experiment_id = "-".join([model, env, training_id])
    results_file_name = experiment_id + ".npy"
    
    if lin_thresh:
        results_file_name = results_file_name.split("-")
        results_file_name[1] += "_linear"
        results_file_name = "-".join(results_file_name)

    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    aggregation_dir = results_dir.joinpath("fixed_env_results",
                                           "-".join(results_file_name.split("-")[:2]))
    results_path = aggregation_dir.joinpath(results_file_name)

    return results_path


def check_lock(results_path: Path) -> bool:
    """
    Checks if the evaluation run defined by the results_path passed needs to be executed, and if so
    checks that there isn't a lock already on it.

    Returns:
        bool: Whether to execute the evaluation run or not.
    """
    lock_name = results_path.name[:-4] + "-fixed_env" + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    if results_path.exists() or lock_path.exists():
        return False

    return True


def acquire_lock(results_path: Path) -> None:
    """
    Adds a lock on the current evaluation run.
    """
    if not LOCKS_PATH.exists():
        LOCKS_PATH.mkdir()
    lock_name = results_path.name[:-4] + "-fixed_env" + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    np.save(lock_path, None)


def release_lock(results_path: Path) -> None:
    """
    Releases the lock on the current evaluation run.
    """
    lock_name = results_path.name[:-4] + "-fixed_env" + ".npy"
    lock_path = LOCKS_PATH.joinpath(lock_name)
    os.remove(lock_path)


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
    elif utility_type == "linear_threshold":
        return [
            ThresholdUtility(thresholds_and_ceofficients=np.array(
                [[0, 0, 0, 0, 0, 0], [-1, -5, r0, r1, r2, r3]], dtype=np.float32))
            for r0 in np.arange(-20, 21, step=2)
            for r1 in np.arange(-20, 21, step=2)
            for r2 in np.arange(-20, 21, step=2)
            for r3 in np.arange(-20, 21, step=2)
            if (r0 > 0) or (r1 > 0) or (r2 > 0) or (r3 > 0)
        ]
    elif utility_type == "target":
        return [
            TargetUtility(target=np.array([31, 31, target0, target1, target2, target3], dtype=np.float32))
            for target0 in range(4) for target1 in range(4) for target2 in range(3)
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

        results.append(np.copy(env._cumulative_rewards))

    return results


def main(_):
    logging.set_verbosity(logging.INFO)

    for env_type, model, training_id, lin_thresh in itertools.product(ENVS, MODELS, TRAINING_IDS, [True, False]):
        if lin_thresh and not "threshold" in env_type:
            continue  # Can't evaluate a non threshold agent on threshold utilities.
        experiment_dir = os.path.join(FLAGS.root_dir, "-".join([model, env_type, training_id]))
        model_path = os.path.join(experiment_dir, 'model', 'dqn_model.h5')
        results_path = generate_results_path(Path(FLAGS.results_dir),
                                             env=env_type,
                                             model=model,
                                             training_id=training_id,
                                             lin_thresh=lin_thresh)

        if not check_lock(results_path=results_path):
            continue

        acquire_lock(results_path=results_path)

        print(f"\n\n{results_path}\n")

        # Loading appropriate gin configs for the environment and this experiment
        gin.clear_config()
        qnet_gin, env_gin = experiment_dir.split("/")[-1].split("-")[:2]
        gin_files = [
            Path("tunable-agents-MORL/configs/envs/fixed_env.gin"),
            Path("tunable-agents-MORL/configs/qnets/" + qnet_gin + ".gin")
        ]
        utility.load_gin_configs(gin_files, [])

        # Loading trained agent model
        env_kwargs = ENV_KWARGS[env_gin]
        if lin_thresh: env_kwargs["utility_type"] = "linear_threshold"
        env = utility.create_environment(**env_kwargs)
        tf_agent = agent.DQNAgent(epsilon=0, obs_spec=env.observation_spec())
        tf_agent.load_model(model_path)

        # Selecting the utilities to run on
        utilities = utility_list(env_kwargs["utility_type"])
        
        # Evaluating the agent on the fixed environment
        results = fixed_env_eval(tf_agent, utilities, **env_kwargs)

        # Save results
        if not results_path.parent.exists():
            results_path.parent.mkdir(parents=True)
        np.save(results_path, results)

        release_lock(results_path=results_path)


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
