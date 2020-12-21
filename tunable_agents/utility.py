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
from tf_agents.environments import tf_py_environment, py_environment
from tunable_agents.environments.DST_env import DST_env
from tunable_agents.environments.gathering_env import gathering_env
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, Flatten, Concatenate

from functools import partial


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.

    Args:
      gin_files: A list of paths to the gin configuration files for this
        experiment.
      gin_bindings: List of gin parameter bindings to override the values in the
        config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


def gathering_replication_agent_qnetwork(obs_spec, action_spec) -> q_network.QNetwork:
    model = tf.keras.Sequential(name='image_encoder')
    model.add(Conv2D(256, (3, 3), activation='relu', name='myconv'))
    model.add(Dropout(0.2, name='mydrop'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='myconv2'))
    model.add(Dropout(0.2, name='mydrop2'))
    model.add(Flatten())

    preprocessing_layers = {'observations': model,
                            'preference_weights': InputLayer(input_shape=tf.TensorShape((6,)), name='PrefInput')}
    
    preprocessing_combiner = Concatenate()
    
    q_net = q_network.QNetwork(
                    obs_spec,
                    action_spec,
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=preprocessing_combiner,
                    fc_layer_params=(64, 64),
                    dropout_layer_params=[0.2, 0.2])
    return q_net


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
def create_agent(
        agent_class: str,
        environment: tf_py_environment.TFPyEnvironment,
        fc_layer_params: Tuple[int],
        learning_rate: float,
        decaying_epsilon: Callable[[], float],
        n_step_update: int,
        target_update_tau: float,
        target_update_period: int,
        gamma: float,
        reward_scale_factor: float,
        gradient_clipping: float,
        debug_summaries: bool,
        summarize_grads_and_vars: bool,
        train_step_counter: tf.Variable,
        num_atoms: int = None,      # Only for categorical_dqn
        min_q_value: float = None,  # Only for categorical_dqn
        max_q_value: float = None,  # Only for categorical_dqn
) -> tf_agent.TFAgent:
    """
    Creates the agent.

    Args:
      agent_class: str, type of agent to construct.
      environment: The environment.
      learning_rate: The Learning Rate
      decaying_epsilon: Epsilon for Epsilon Greedy Policy
      target_update_tau: Agent parameter
      target_update_period: Agent parameter
      gamma: Agent parameter
      reward_scale_factor: Agent parameter
      gradient_clipping: Agent parameter
      debug_summaries: Agent parameter
      summarize_grads_and_vars: Agent parameter
      train_step_counter: The train step tf.Variable to be passed to agent


    Returns:
      An agent for playing Hanabi.

    Raises:
      ValueError: if an unknown agent type is requested.
    """
    if agent_class == 'DQN':
        return dqn_agent.DqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_network.QNetwork(
                environment.time_step_spec().observation['observations'],
                environment.action_spec(),
                fc_layer_params=fc_layer_params),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=
            observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    elif agent_class == 'DDQN':
        return dqn_agent.DdqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_network.QNetwork(
                environment.time_step_spec().observation['observations'],
                environment.action_spec(),
                fc_layer_params=fc_layer_params),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=
            observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    elif agent_class == 'categorical_dqn':
        return categorical_dqn_agent.CategoricalDqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            categorical_q_network=categorical_q_network.CategoricalQNetwork(
                environment.time_step_spec().observation['observations'],
                environment.action_spec(),
                num_atoms=num_atoms,
                fc_layer_params=fc_layer_params),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            observation_and_action_constraint_splitter=
            observation_and_action_constraint_splitter,
            epsilon_greedy=decaying_epsilon,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    elif agent_class == 'replication_study':
        q_net = gathering_replication_agent_qnetwork(environment.time_step_spec().observation,
                                                     environment.action_spec())
        return dqn_agent.DqnAgent(
            environment.time_step_spec(),
            environment.action_spec(),
            q_network=q_net,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            epsilon_greedy=decaying_epsilon,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)
    else:
        raise ValueError(
            'Expected valid agent_type, got {}'.format(agent_class))


@gin.configurable(denylist=['data_spec', 'batch_size'])
def create_replay_buffer(data_spec, batch_size: int, max_length: int) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec, batch_size=batch_size, max_length=max_length)


@gin.configurable
def no_decay(epsilon: float) -> float:
    return epsilon


@gin.configurable(denylist=['step'])
def linear_decay(initial_epsilon: float,
                 final_epsilon: float,
                 step: tf.Variable,
                 decay_time: int) -> float:
    """
    Linear decay from initial_epsilon to final_epsilon in the given decay_time as measured by step.
    It is assumed that initial_epsilon > final_epsilon.
    """
    return tf.cast(initial_epsilon - (initial_epsilon - final_epsilon)*(step/decay_time), dtype=tf.float32)


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
    
    return (0.5 ** tf.cast((step // decay_time), tf.float32)) * initial_epsilon


@gin.configurable(denylist=['step'])
def decaying_epsilon(step: tf.Variable,
                     decay_type: str = 'exponential'
                     ) -> Callable[[], float]:

    if decay_type == 'exponential':
        return partial(exponential_decay, step=step)
    elif decay_type == 'linear':
        return partial(linear_decay, step=step)
    elif decay_type == None:
        return partial(no_decay)
    else:
        raise NotImplementedError(
            'decay_type requested is not implemented yet.')


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
