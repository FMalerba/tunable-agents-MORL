from typing import Callable, Tuple, Union
import gin.tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
import tensorflow as tf

Agent = Union[dqn_agent.DqnAgent, dqn_agent.DdqnAgent, categorical_dqn_agent.CategoricalDqnAgent]

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


@gin.configurable
def create_environment(game_type='Hanabi-Full'):
    """Creates the environment.

	Args:
	  game_type: Type of game to play. Currently the following are supported:
		Hanabi-Full: Regular game.
		Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.

	Returns:
		A PyEnvironment object environment.
	"""
    raise NotImplementedError('Still have to do this')


@gin.configurable(blacklist=['environment', 'train_step_counter'])
def create_agent(
        agent_class: str,
        environment: tf_py_environment.TFPyEnvironment,
        fc_layer_params: Tuple[int],
        learning_rate: float,
        decaying_epsilon: Callable[[], float],
        n_step_update,
        target_update_tau,
        target_update_period,
        gamma,
        reward_scale_factor,
        gradient_clipping,
        debug_summaries,
        summarize_grads_and_vars,
        train_step_counter,
        num_atoms=None,  # Only for categorical_dqn
        min_q_value=None,  # Only for categorical_dqn
        max_q_value=None,  # Only for categorical_dqn
) -> Agent:
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
    else:
        raise ValueError(
            'Expected valid agent_type, got {}'.format(agent_class))


@gin.configurable(blacklist=['data_spec', 'batch_size'])
def create_replay_buffer(data_spec, batch_size: int, max_length: int):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=data_spec, batch_size=batch_size, max_length=max_length),



@gin.configurable(blacklist=['train_step'])
def decaying_epsilon(initial_epsilon: float,
                     train_step,
                     decay_time: int,
                     decay_type: Union[str, None] = 'exponential',
                     reset_at_step: int = None) -> float:
    # TODO Implementing an automatic reset of the decaying epsilon parameter? Something maybe that looks at the variance
    # of some performance metric(s) and decides based on that if it should reset the decay of epsilon or not
    if reset_at_step:
        # The reason why these two ifs are separated and not grouped in an *and* expression (using python short-circuit)
        # is because this function might actually get optimized by tf.function since it's called inside the agent training function
        # and I think short-circuiting doesn't work in graph mode.
        if reset_at_step <= train_step:
            # Notice that this doesn't change the train_step outside the scope of this function
            # (which is the desired behaviour)
            train_step = train_step - reset_at_step
    if decay_type == 'exponential':
        decay = 0.5**tf.cast((train_step // decay_time), tf.float32)
    elif decay_type == None:
        decay = 1
    else:
        raise NotImplementedError(
            'Only exponential decay and no decay are implmented for now.')

    return initial_epsilon * decay


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
