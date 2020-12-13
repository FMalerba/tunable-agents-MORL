from tunable_agents_environment import utility_functions
from collections.abc import Callable
from typing import Tuple, List, Union
import numpy as np

GRID_ROWS = 11
GRID_COLS = 10

TREASURE_DEPTHS = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
TREASURE_VALUES = [1, 2, 3, 5, 8, 16, 24, 50, 74, 124]
UTILITY_FUNCTIONS = (utility_functions.linear_utility, utility_functions.polinomial_utility, utility_functions.threshold_utility)

# UP, DOWN, LEFT, RIGHT
ACTIONS = [[-1,0],[1,0],[0,-1],[0,1]]

class DeepSeaTreasureEnvironment:
    def __init__(self):
        self._forbidden_states = [(i, j) for j in range(GRID_COLS) for i in range(TREASURE_DEPTHS[j]+1, GRID_ROWS)]
        self._treasure_locations = set([(i, j) for j, i in enumerate(TREASURE_DEPTHS)])
        self._utility_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    
    
    def reset(self):
        self.n_steps = 0
        self.state = [0, 0]
        self._utility_function = np.random.choice(UTILITY_FUNCTIONS)
        return self.state
    
    
    def step(self, action: int):
        """
        Transition the environment through the input action
        """
        self.n_steps += 1
        
        self.state[0] += ACTIONS[action][0]
        self.state[1] += ACTIONS[action][1]
        
        legal_actions = self._legal_actions()
        rewards = self._get_rewards()
        state = self.state
        done = self._is_terminal()
        return state, rewards, done
    
    
    def _legal_actions(self) -> List[bool]:
        #TODO Define legal actions in the correct format for the agent.
        if self.state[1] == 6 and self.state[0] in (5, 6):
            return
        if self.state[1] == 8 and self.state[0] == 8:
            return
        
        return
    
    
    def _get_rewards(self):
        treasure_cond = self.state in self._treasure_locations
        return (-1, TREASURE_VALUES[self.state[1]] if treasure_cond else 0)         # (time_penalty, treasure_reward)
    
    
    def _is_terminal(self):
        return (self.state in self._treasure_locations) or (self.n_steps > 200)


class DST_wrapper():
    def __init__(self) -> None:
        super().__init__()