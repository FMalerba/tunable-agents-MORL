from gym_mo.envs.gridworlds import mo_gathering_env
from tunable_agents_environment import utility_functions
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import gin.tf

from tf_agents.typing import types
from collections.abc import Callable
from typing import Tuple, List, Union

import numpy as np



class ObservationStacker(object):
    """Class for stacking agent observations."""

    def __init__(self, history_size: int, observation_size: int, num_players):
        """Initializer for observation stacker.

        Args:
          history_size: int, number of time steps to stack.
          observation_size: int, size of observation vector on one time step.
          num_players: int, number of players.
        """
        self._history_size = history_size
        self._observation_size = observation_size
        self._num_players = num_players
        self._obs_stacks = list()
        for _ in range(0, self._num_players):
            self._obs_stacks.append(np.zeros(self._observation_size *
                                             self._history_size))

    def add_observation(self, observation, current_player):
        """Adds observation for the current player.

        Args:
          observation: observation vector for current player.
          current_player: int, current player id.
        """
        self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                                   -self._observation_size)
        self._obs_stacks[current_player][(self._history_size - 1) *
                                         self._observation_size:] = observation

    def get_observation_stack(self, current_player):
        """Returns the stacked observation for current player.

        Args:
          current_player: int, current player id.
        """

        return self._obs_stacks[current_player]

    def reset_stack(self):
        """Resets the observation stacks to all zero."""

        for i in range(0, self._num_players):
            self._obs_stacks[i].fill(0.0)

    @property
    def history_size(self):
        """Returns number of steps to stack."""
        return self._history_size

    def observation_size(self):
        """Returns the size of the observation vector after history stacking."""
        return self._observation_size * self._history_size




@gin.configurable(blacklist=['environment'])
def create_obs_stacker(environment: py_environment.PyEnvironment, history_size: int = 4):
    """Creates an observation stacker.

    Args:
      environment: environment object.
      history_size: int, number of steps to stack.

    Returns:
      An observation stacker object.
    """

    return ObservationStacker(history_size,
                              environment.observation_spec().shape()[0],
                              num_players=1)




@gin.configurable
class Gathering_wrapper(py_environment.PyEnvironment):
    def __init__(self, preference: np.ndarray = None,
                 gamma: float = 0.99, history_size: int = 3) -> None:
        super().__init__()
        # TODO make the environment so that if a preference vector is passed to the wrapper
        # then the preferences are fixed by it. If it's not passed, a new preference vector is
        # sampled every new episode (reset call). The MO-env under the wrapper already accepts
        # a prefernce vector as input in its reset, I just need to modify the wrapper's _reset to
        # sample a preference vector like Daniel and push it down
        self._preference = preference
        self._env = mo_gathering_env.MOGatheringEnv()
        self.gamma = gamma
        self._obs_stacker = create_obs_stacker(self, history_size=history_size)
        self._observation_spec = {'observations': array_spec.ArraySpec(shape=(2,), dtype=np.int64),
                                  'weights': array_spec.ArraySpec(shape=(2,), dtype=np.int64)}
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int_, minimum=0, maximum=4)
    
    
    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec()
    
    
    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec()
    
    
    def _reset(self) -> ts.TimeStep:
        if self._preference:
            obs, legal_moves = self._env.reset(self._preference)
        else:
            obs, legal_moves = self._env.reset(self._sample_preference)
        observations_and_legal_moves = {'observations': obs,
                                        'legal_moves': legal_moves}
        return ts.restart(observations_and_legal_moves)
    
    
    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._current_time_step.is_last():
            return self.reset()
        
        obs, rewards, done = self._env.step(action)
        
        observations_and_legal_moves = {'observations': obs,
                                        'legal_moves': legal_moves}
        
        # TODO Manage the reward-utility
        if done:
            return ts.termination(observations_and_legal_moves, reward)
        else:
            return ts.transition(observations_and_legal_moves, reward, self.gamma)
        
