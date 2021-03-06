from typing import Tuple
from gym_mo.envs.gridworlds.mo_gridworld_base import MOGridworld
from tunable_agents.environments.utility_functions import sample_utility, DEFAULT_SAMPLING
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import gin.tf

from tf_agents.typing import types

import numpy as np

# Before adding new utilities to this list check their behaviour in particular for the
# GatheringWrapper._is_done() method
SUPPORTED_UTILITY_TYPES = [
    "linear", "threshold", "linear_threshold", "dual_threshold", "linear_dual_threshold", "target"
]


class ObservationStacker(object):
    """
    Class for stacking agent observations.
    Observations are stacked along their last dimension; i.e. three 8x8 rgb images (8x8x3)
    would result in a 8x8x9 observation stack.
    """

    def __init__(self, history_size: int, single_observation_shape: Tuple[int]):
        """Initializer for observation stacker.

        Args:
          history_size: int, number of time steps to stack.
          single_observation_shape: tuple of integers, shape of observation vector on one time step.
        """
        self._history_size = history_size
        self._single_observation_shape = single_observation_shape
        self._obs_stack: np.ndarray = np.zeros(shape=(*single_observation_shape[:-1],
                                                      single_observation_shape[-1] * history_size),
                                               dtype=np.float32)

    def add_observation(self, observation: np.ndarray):
        """Adds observation for the current player.

        Args:
          observation: observations to be added to the stack.
        """
        self._obs_stack = np.roll(self._obs_stack, -self._single_observation_shape[-1], axis=-1)
        self._obs_stack[..., -self._single_observation_shape[-1]:] = observation

    def get_observation_stack(self):
        """Returns the stacked observations."""
        return self._obs_stack

    def reset_stack(self):
        """Resets the observation stacks to all zeroes"""
        self._obs_stack.fill(0.0)

    @property
    def history_size(self):
        """Returns number of steps to stack."""
        return self._history_size

    def stacked_observation_shape(self):
        """Returns the shape of the observations after history stacking."""
        return self._obs_stack.shape


@gin.configurable(denylist=['environment'])
def create_obs_stacker(environment: py_environment.PyEnvironment, history_size: int = 3):
    """Creates an observation stacker.

    Args:
      environment: Gathering object.
      history_size: int, number of steps to stack.

    Returns:
      An observation stacker object.
    """

    return ObservationStacker(history_size, environment.single_obs_shape())


@gin.configurable()
class GatheringWrapper(py_environment.PyEnvironment):

    def __init__(self,
                 cumulative_rewards_flag: bool = False,
                 gamma: float = 1,
                 history_size: int = 3,
                 utility_repr: np.ndarray = None,
                 utility_type: str = 'linear',
                 sampling: str = DEFAULT_SAMPLING,
                 **gridworld_kwargs) -> None:
        super().__init__()
        if utility_type not in SUPPORTED_UTILITY_TYPES:
            raise ValueError(
                "{} is not among the yet supported utility types for the gathering environment".format(utility_type))

        # If a utility representation is passed to the environment, then the corresponding utility is fixed and won't be resampled
        self._fixed_utility = utility_repr is not None
        self._utility_type = utility_type
        self._sampling = sampling
        self._utility_func = sample_utility(utility_type=utility_type,
                                            sampling=sampling,
                                            utility_repr=utility_repr)
        self._interests = self._utility_func.interests

        self._cumulative_rewards_flag = cumulative_rewards_flag
        self._cumulative_rewards: np.ndarray = np.zeros(shape=(6,), dtype=np.float32)

        self._env = MOGridworld(**gridworld_kwargs)
        self.gamma = gamma
        self._obs_stacker = create_obs_stacker(self, history_size=history_size)

        self._observation_spec = {
            'state_obs':
                array_spec.ArraySpec(shape=self._obs_stacker.stacked_observation_shape(), dtype=np.float32),
            'utility_representation':
                array_spec.ArraySpec(shape=self._utility_func.agent_utility_repr.shape, dtype=np.float32)
        }
        if cumulative_rewards_flag:
            self._observation_spec['cumulative_rewards'] = array_spec.ArraySpec(shape=(6,), dtype=np.float32)
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int_, minimum=0, maximum=4)

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def single_obs_shape(self) -> Tuple[int]:
        return (8, 8, 3)

    def _reset(self) -> ts.TimeStep:
        if not self._fixed_utility:
            self._utility_func = sample_utility(utility_type=self._utility_type, sampling=self._sampling)
            self._interests = self._utility_func.interests

        state_obs = self._env.reset()

        self._cumulative_rewards.fill(0.0)
        self._prev_step_utility = self._utility_func(self._cumulative_rewards)

        self._obs_stacker.reset_stack()
        self._obs_stacker.add_observation(state_obs / 255)  # Normalizing obs in range [0, 1]
        stacked_obs = self._obs_stacker.get_observation_stack()

        obs = {
            'state_obs': stacked_obs,
            'utility_representation': self._utility_func.agent_utility_repr,
        }
        if self._cumulative_rewards_flag:
            obs['cumulative_rewards'] = self._cumulative_rewards

        return ts.restart(obs)

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._current_time_step.is_last():
            return self.reset()

        state_obs, rewards, _ = self._env.step(action)
        done = self._is_done()
        self._cumulative_rewards += rewards

        self._obs_stacker.add_observation(state_obs / 255)  # Normalizing obs in range [0, 1]
        stacked_obs = self._obs_stacker.get_observation_stack()

        obs = {
            'state_obs': stacked_obs,
            'utility_representation': self._utility_func.agent_utility_repr,
        }
        if self._cumulative_rewards_flag:
            obs['cumulative_rewards'] = self._cumulative_rewards

        # The scalar reward on which to train is equal to the delta in the utility between the
        # previous time step and the current one.
        current_utility = self._utility_func(self._cumulative_rewards)
        reward = current_utility - self._prev_step_utility
        self._prev_step_utility = current_utility

        if done:
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, self.gamma)

    def _is_done(self) -> bool:
        if self._env.step_count > self._env.max_steps:
            return True

        if self._utility_type in ["target", "dual_threshold", "linear_dual_threshold"]:
            interests = self._interests - self._cumulative_rewards
        else:
            interests = self._interests

        for row in range(self._env.grid.shape[0]):
            for column in range(self._env.grid.shape[1]):
                if self._env.grid[row, column] is not None and interests[self._env.grid[row, column].idx] > 0:
                    return False

        for agent in self._env.agents:
            if (interests[agent.idx] > 0) and (not agent.is_done):
                return False

        return True
