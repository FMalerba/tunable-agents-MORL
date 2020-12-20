from typing import Tuple
from gym_mo.envs.gridworlds import mo_gathering_env
from tunable_agents.environments import utility_functions
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import gin.tf

from tf_agents.typing import types

import numpy as np



class ObservationStacker(object):
    """
    Class for stacking agent observations.
    Observations are stacked along their last dimension; i.e. three 8x8 rgb images (8x8x3)
    would result in a 8x8x9 observation stack.
    """

    def __init__(self, history_size: int, observation_shape: Tuple[int]):
        """Initializer for observation stacker.

        Args:
          history_size: int, number of time steps to stack.
          observation_size: tuple of integers, shape of observation vector on one time step.
        """
        self._history_size = history_size
        self._observation_shape = observation_shape
        self._obs_stack : np.ndarray = np.zeros(shape=(*observation_shape[:-1], observation_shape[-1]*history_size), dtype=np.float32)

    def add_observation(self, observation: np.ndarray):
        """Adds observation for the current player.

        Args:
          observation: observations to be added to the stack.
        """
        self._obs_stack = np.roll(self._obs_stack, -self._observation_shape[-1], axis=-1)
        self._obs_stack[..., -self._observation_shape[-1]:] = observation

    def get_observation_stack(self):
        """Returns the stacked observations."""
        return self._obs_stack

    def reset_stack(self):
        """Resets the observation stacks to all zero."""
        self._obs_stack.fill(0.0)

    @property
    def history_size(self):
        """Returns number of steps to stack."""
        return self._history_size

    def observation_shape(self):
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

    return ObservationStacker(history_size,
                              environment.single_obs_shape())



@gin.configurable
class GatheringWrapper(py_environment.PyEnvironment):
    def __init__(self, preference: np.ndarray = None,
                 gamma: float = 0.99, history_size: int = 3) -> None:
        super().__init__()
        # If a preference is passed to the environment, then such preference is fixed and won't be resampled
        self._fixed_preference = bool(preference)
        self._preference = preference
        self._env = mo_gathering_env.MOGatheringEnv()
        self.gamma = gamma
        self._obs_stacker = create_obs_stacker(self, history_size=history_size)
        self._observation_spec = {'observations': array_spec.ArraySpec(shape=self._obs_stacker.observation_shape(), dtype=np.float32),
                                  'preference_weights': array_spec.ArraySpec(shape=(6,), dtype=np.float32)}
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int_, minimum=0, maximum=4)
    
    
    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec
    
    
    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec
    
    
    def single_obs_shape(self) -> Tuple[int]:
        return (8,8,3)
    
    
    def _reset(self) -> ts.TimeStep:
        if self._fixed_preference:
            obs = self._env.reset(self._preference)
        else:
            self._preference = utility_functions.sample_preference()
            obs = self._env.reset()
        
        self._obs_stacker.reset_stack()
        self._obs_stacker.add_observation(obs)
        stacked_obs = self._obs_stacker.get_observation_stack()
        
        observations_and_preferences = {'observations': stacked_obs,
                                        'preference_weights': self._preference/40}
        return ts.restart(observations_and_preferences)
    
    
    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._current_time_step.is_last():
            return self.reset()
        
        obs, rewards, done, _ = self._env.step(action)
        
        self._obs_stacker.add_observation(obs)
        stacked_obs = self._obs_stacker.get_observation_stack()
        
        # Preferences are in range [-20, 20] we normalize them to the range [-0.5, 0.5]
        observations_and_preferences = {'observations': stacked_obs,
                                        'preference_weights': self._preference/40}
        
        reward = np.dot(rewards, self._preference)
        if done:
            return ts.termination(observations_and_preferences, reward)
        else:
            return ts.transition(observations_and_preferences, reward, self.gamma)
    
