from functools import partial
import gin.tf
from typing import Callable

import tensorflow as tf

from tf_agents.environments import py_environment

from tunable_agents.environments.DST_env import DST_env
from tunable_agents.environments.gathering_env import gathering_env
from tunable_agents import external_configurables  # Don't remove, it's necessary to configure TF Layers


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
      gin_files: A list of paths to the gin configuration files for this
        experiment.
      gin_bindings: List of gin parameter bindings to override the values in the
        config files.
    """
    try:
        path_folders = gin_files[0].split('/')
        configs_folder_index = path_folders.index('configs')
    except:
        raise ValueError("Expected gin_files paths to be like {}, instead got {}".format('.../configs/...', gin_files[0]))
    
    configs_folder_path = '/'.join(path_folders[:configs_folder_index + 1])
    gin.add_config_file_search_path(configs_folder_path)
    gin.parse_config_files_and_bindings(gin_files, bindings=gin_bindings, skip_unknown=False)


@gin.configurable
def create_environment(game: str = 'gathering') -> py_environment.PyEnvironment:
    """Creates the environment.
    
    Args:
        game: Game to be played
    
    Returns:
    A PyEnvironment object environment.
    """
    if game == 'DST':
        return DST_env.DSTWrapper()
    elif game == 'gathering':
        return gathering_env.GatheringWrapper()
    raise NotImplementedError('Game is not among the implemented games')


@gin.configurable
def no_decay(epsilon: float) -> float:
    return epsilon


@gin.configurable(denylist=['step'])
def linear_decay(initial_epsilon: float, final_epsilon: float, step: tf.Variable, decay_time: int) -> float:
    """
    Linear decay from initial_epsilon to final_epsilon in the given decay_time as measured by step.
    It is assumed that initial_epsilon > final_epsilon.
    """
    return tf.cast(tf.maximum(initial_epsilon - (initial_epsilon - final_epsilon) * (step / decay_time),
                              final_epsilon),
                   dtype=tf.float32)


@gin.configurable(denylist=['step'])
def exponential_decay(initial_epsilon: float,
                      step: tf.Variable,
                      decay_time: int,
                      reset_at_step: int = 0) -> float:
    # TODO Implementing an automatic reset of the decaying epsilon parameter? Something maybe that looks at the variance
    # of some performance metric(s) and decides based on that if it should reset the decay of epsilon or not
    if reset_at_step <= step:
        # Notice that this doesn't change the train_step outside the scope of this function
        # (which is the desired behaviour)
        step = step - reset_at_step

    return (0.5**tf.cast((step // decay_time), tf.float32)) * initial_epsilon


@gin.configurable(denylist=['step'])
def decaying_epsilon(step: tf.Variable, decay_type: str = 'exponential') -> Callable[[], float]:

    if decay_type == 'exponential':
        return partial(exponential_decay, step=step)
    elif decay_type == 'linear':
        return partial(linear_decay, step=step)
    elif decay_type == None:
        return partial(no_decay)
    else:
        raise NotImplementedError('decay_type requested is not implemented yet.')

