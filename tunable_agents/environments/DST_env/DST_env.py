from tunable_agents.environments import utility_functions
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import gin.tf

from tf_agents.typing import types
from collections.abc import Callable
from typing import Tuple, List

import numpy as np

GRID_ROWS = 11
GRID_COLS = 10

TREASURE_DEPTHS = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
TREASURE_VALUES = [1., 2., 3., 5., 8., 16., 24., 50., 74., 124.]

# UP, DOWN, LEFT, RIGHT
ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]


class DeepSeaTreasureEnvironment:

    def __init__(self):
        self._forbidden_states = [
            (i, j) for j in range(GRID_COLS) for i in range(TREASURE_DEPTHS[j] + 1, GRID_ROWS)
        ]
        self._treasure_locations = set([(i, j) for j, i in enumerate(TREASURE_DEPTHS)])
        self._utility_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = None

    def reset(self) -> Tuple[List[int], Tuple[bool]]:
        self._n_steps = 0
        self._state = [0, 0]
        self._utility_function = utility_functions.generate_utility()
        legal_moves = self._legal_moves()
        return self._state, legal_moves

    def step(self, action: int) -> Tuple[List[int], Tuple[bool], Tuple[float], bool]:
        """
        Transition the environment through the input action.
        Arg:
            action: an integer in [0, 3] that identifies the action to be taken. Must be a legal move.
        Returns:

        """
        self._n_steps += 1

        self._state[0] += ACTIONS[action][0]
        self._state[1] += ACTIONS[action][1]

        legal_moves = self._legal_moves()
        rewards = self._get_rewards()
        obs = self._state
        done = self._is_terminal()
        return obs, legal_moves, rewards, done

    def _legal_moves(self) -> Tuple[bool]:
        """
        Returns a 4-tuple of bool values specifying for every action (Up, Down, Left, Right)
        if it's valid (True) or not (False).
        """
        if self._state[0] == 0:
            # Can't go up
            if self._state[1] == 0:
                # Can't go left
                return (False, True, False, True)
            if self._state[1] == GRID_COLS - 1:
                # Can't go right
                return (False, True, True, False)
            return (False, True, True, True)

        if self._state[1] == GRID_COLS - 1:
            # Can't go right
            return (True, True, True, False)

        if (self._state[1] == 6 and self._state[0] in (5, 6)) or (self._state[1] == 8 and
                                                                  self._state[0] == 8):
            # Can't go left because of sea floor
            return (True, True, False, True)

        return (True, True, True, True)

    def _get_rewards(self) -> Tuple[float]:
        treasure_cond = self._state in self._treasure_locations
        return (-1., TREASURE_VALUES[self._state[1]] if treasure_cond else 0
                )  # (time_penalty, treasure_reward)

    def _is_terminal(self) -> bool:
        return (self._state in self._treasure_locations) or (self._n_steps > 200)


@gin.configurable
class DSTWrapper(py_environment.PyEnvironment):

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self._env = DeepSeaTreasureEnvironment()
        self.gamma = gamma
        self._observation_spec = {
            'observations': array_spec.ArraySpec(shape=(2,), dtype=np.int64),
            'legal_moves': array_spec.ArraySpec(shape=(4,), dtype=np.bool_)
        }
        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int_,
                                                        minimum=0,
                                                        maximum=self.num_moves() - 1)

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _reset(self) -> ts.TimeStep:
        obs, legal_moves = self._env.reset()
        observations_and_legal_moves = {'observations': obs, 'legal_moves': legal_moves}
        return ts.restart(observations_and_legal_moves)

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._current_time_step.is_last():
            return self.reset()

        obs, legal_moves, rewards, done = self._env.step(action)

        observations_and_legal_moves = {'observations': obs, 'legal_moves': legal_moves}

        # TODO Manage the reward-utility
        if done:
            return ts.termination(observations_and_legal_moves, reward)
        else:
            return ts.transition(observations_and_legal_moves, reward, self.gamma)
