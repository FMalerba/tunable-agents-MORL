from typing import Callable, Tuple
import gin.tf
from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.environments import tf_py_environment, py_environment
from tunable_agents.environments.DST_env import DST_env
from tunable_agents.environments.gathering_env import gathering_env
from tunable_agents import external_configurables  # Necessary to configure TF Layers
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, InputLayer
from tensorflow.python.keras.engine.sequential import Sequential

from functools import partial


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
def create_preprocessing(**kwargs) -> Sequential:
    """
    Creates preprocessing layers for a single input using tf.keras.Sequential
    
    This function is to only be called inside the gin-config files and serves as a 
    wrapper to config Sequential.
    """
    """
    The reason for not using gin.config.external_configurable is that the Sequential's 
    configurations seem to leak to other Sequential calls. Specifically, EncodingNetwork
    copies the preprocessing_layers by creating a new Sequential object to which are then added
    all the layers. The fact that configs leak, raises an error because the new Sequential is
    spawned with all the layers from the configuration, and then the program tries to add them
    again raising a NameError because you get two layers with the same name.
    """

    return Sequential(**kwargs)


@gin.configurable
def create_qnet(obs_spec: types.Spec, action_spec: types.Spec,
                preprocessing_layers: dict) -> q_network.QNetwork:
    # Any observation that don't have already a defined preprocessing are simply
    # fed to an InputLayer to then be concatenated with the output of other preprocessing
    for obs in obs_spec:
        if obs not in preprocessing_layers:
            preprocessing_layers[obs] = InputLayer(input_shape=obs_spec[obs].shape,
                                                   name='{}_Input'.format(obs))

    preprocessing_combiner = Concatenate()

    return q_network.QNetwork(obs_spec,
                              action_spec,
                              preprocessing_layers=preprocessing_layers,
                              preprocessing_combiner=preprocessing_combiner)


@gin.configurable
def create_environment(game: str = 'DST') -> py_environment.PyEnvironment:
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


@gin.configurable(denylist=['environment', 'train_step_counter'])
def create_agent(agent_class: str, environment: tf_py_environment.TFPyEnvironment, learning_rate: float,
                 decaying_epsilon: Callable[[], float], train_step_counter: tf.Variable) -> tf_agent.TFAgent:
    """
    Creates the agent.

    Args:
      agent_class: str, type of agent to construct
      environment: The environment
      learning_rate: The Learning Rate
      decaying_epsilon: Epsilon for Epsilon Greedy Policy
      n_step_update: The number of steps for the policy updateS
      train_step_counter: The train step tf.Variable to be passed to agent


    Returns:
      An agent from TF-Agents

    Raises:
      ValueError: if an unknown agent type is requested.
    """
    if agent_class == 'DQN':
        obs_spec = environment.time_step_spec().observation
        obs_spec.pop('legal_moves', None)  # Pop legal moves if there are any
        q_net = create_qnet(obs_spec, environment.action_spec())
        return dqn_agent.DqnAgent(environment.time_step_spec(),
                                  environment.action_spec(),
                                  q_network=q_net,
                                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                  epsilon_greedy=decaying_epsilon,
                                  td_errors_loss_fn=common.element_wise_squared_loss,
                                  train_step_counter=train_step_counter)
    elif agent_class == 'DDQN':
        return dqn_agent.DdqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_network.QNetwork(environment.time_step_spec().observation['observations'],
                                         environment.action_spec()),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
    elif agent_class == 'categorical_dqn':
        return categorical_dqn_agent.CategoricalDqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            categorical_q_network=categorical_q_network.CategoricalQNetwork(
                environment.time_step_spec().observation['observations'], environment.action_spec()),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
    else:
        raise ValueError('Expected valid agent_type, got {}'.format(agent_class))


@gin.configurable(denylist=['data_spec', 'batch_size'])
def create_replay_buffer(data_spec, batch_size: int,
                         max_length: int) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
    # TODO pass the GPU as device parameter to the RB in order to store it in VRAM.
    # Analyse performance improvements that this might bring.
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=data_spec,
                                                          batch_size=batch_size,
                                                          max_length=max_length)


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


def observation_and_action_constraint_splitter(obs) -> Tuple:
    return obs['observations'], obs['legal_moves']


def print_readable_timestep(time_step: ts.TimeStep, environment: tf_py_environment.TFPyEnvironment):
    # TODO This is a direct copy from a completely different project and won't work as is.
    # Needs to be adapted to the new problem setting.
    obs = time_step.observation["observations"][0]
    print('Last reward:', time_step.reward.numpy())
    for i in range(environment._env.num_moves()):
        if time_step.observation["legal_moves"].numpy()[0][i]:
            print(environment._env.game.get_move(i), ' - ', i)
