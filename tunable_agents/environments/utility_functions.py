from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class UtilityFunction(ABC):

    def __init__(self, utility_repr: np.ndarray, agent_utility_repr: np.ndarray,
                 gridworld_utility_repr: np.ndarray) -> None:
        """
        There are two utility representations, one that is fed as input to the agent and the other that is fed as preference
        representation to the gridworld object underlying the GatheringWrapper environment.
        """
        self._utility_repr = utility_repr
        self._agent_utility_repr = agent_utility_repr
        self._gridworld_utility_repr = gridworld_utility_repr

    def __call__(self, rewards: np.ndarray) -> np.ndarray:
        return self.call(rewards)

    @abstractmethod
    def call(self, rewards: np.ndarray) -> np.ndarray:
        """Implements the utility function"""

    @property
    def utility_repr(self):
        return self._utility_repr

    @property
    def agent_utility_repr(self):
        return self._agent_utility_repr

    @property
    def gridworld_utility_repr(self):
        return self._gridworld_utility_repr


class LinearUtility(UtilityFunction):

    def __init__(self,
                 weights: Optional[np.ndarray] = None,
                 agent_utility_repr: Optional[np.ndarray] = None,
                 gridworld_utility_repr: Optional[np.ndarray] = None) -> None:
        """
        Contains all information and possible different representations of a utility function.
        
        At least one of the three optional arguments must be provided in order to reconstruct the others.
        """
        if weights is not None:
            agent_utility_repr = weights[2:] / 40
            gridworld_utility_repr = weights
        elif agent_utility_repr is not None:
            weights = np.concatenate(([-1, -5], agent_utility_repr * 40)).astype(np.float32)
            gridworld_utility_repr = weights
        elif gridworld_utility_repr is not None:
            weights = gridworld_utility_repr
            agent_utility_repr = weights[2:] / 40
        else:
            raise ValueError(
                "Expected to receive at least one utility representation argument, but received none.")

        self._weights = weights

        super().__init__(utility_repr=weights,
                         agent_utility_repr=agent_utility_repr,
                         gridworld_utility_repr=gridworld_utility_repr)

    def call(self, rewards: np.ndarray) -> np.ndarray:
        return np.dot(rewards, self._weights)


class ThresholdUtility(UtilityFunction):

    def __init__(self,
                 thresholds_and_ceofficients: Optional[np.ndarray] = None,
                 agent_utility_repr: Optional[np.ndarray] = None,
                 gridworld_utility_repr: Optional[np.ndarray] = None) -> None:
        """
        Contains all information and possible different representations of a utility function.
        
        It is required that either thresholds_and_ceofficients is provided or the agent_utility_repr in order to reconstruct the others.
        """
        if thresholds_and_ceofficients is not None:
            thresholds = thresholds_and_ceofficients[0]
            coefficients = thresholds_and_ceofficients[1]
            agent_utility_repr = (thresholds_and_ceofficients[:, 2:] / [[1], [40]]).flatten()
            gridworld_utility_repr = thresholds_and_ceofficients[1]
        elif agent_utility_repr is not None:
            thresholds = np.concatenate(([0, 0], agent_utility_repr[:4])).astype(np.float32)
            coefficients = np.concatenate(([-1, -5], agent_utility_repr[4:] * 40)).astype(np.float32)
            gridworld_utility_repr = coefficients
        else:
            raise ValueError(
                "Expected to receive at least one utility representation argument, but received none.")

        self._thresholds = thresholds
        self._coefficients = coefficients
        super().__init__(utility_repr=np.array([thresholds, coefficients]),
                         agent_utility_repr=agent_utility_repr,
                         gridworld_utility_repr=gridworld_utility_repr)

    def call(self, rewards: np.ndarray) -> np.ndarray:
        return np.sum(np.where(rewards >= self._thresholds, rewards * self._coefficients, 0))


def polinomial_utility(reward: np.ndarray, coefficients: np.ndarray) -> float:
    poly = np.expand_dims(reward, axis=1).repeat(coefficients.shape[1],
                                                 axis=1)**np.arange(coefficients.shape[1])
    return np.sum(poly * coefficients)


def sample_linear_weights() -> np.ndarray:
    """
    Samples a 6-long vector of preferences for the gathering environment replication study.
    Each preference weight is randomly sampled between -20 and 20 in steps of 5 with the 
    exception of the first two entries which are fixed at -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall
    """
    # The 4 entries of pref are the preference for (respectively): Green, Red, Yellow, Other agent taking red
    pref = np.random.choice(np.arange(-20, 21, step=5, dtype=np.float32), size=4)
    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(pref <= 0):
        return sample_linear_weights()

    w01 = np.array([-1, -5], dtype=np.float32)
    return np.concatenate((w01, pref))


def sample_thresholds_and_coefficients() -> np.ndarray:
    """
    Samples a 6-long vector of thresholds for the gathering environment and a 6-long 
    vector of coefficients to be applied once the thresholds are exceeded.
    All thresholds and coefficients are sampled at random with the exception of the first two entries
    which are fixed at a threshold of 0 and coefficients of  -1 and -5. They respectively signal
    the punishment for taking a further time-step and the punishment for hitting a wall.
    """
    # The 4 entries of thresholds and coefficients are for (respectively): Green, Red, Yellow, Other agent taking red
    thresholds = np.random.choice(np.arange(0, 4, dtype=np.float32), size=4)
    coefficients = np.random.choice(np.arange(-20, 21, step=5, dtype=np.float32), size=4)

    # An environment with a negative preference vector will simply stop the episode after the first step.
    # It is therefore pointless to sample such a vector.
    if np.all(coefficients <= 0):
        return sample_thresholds_and_coefficients()

    w01 = np.array([-1, -5], dtype=np.float32)
    return np.array(
        [np.concatenate(([0, 0], thresholds)).astype(np.float32),
         np.concatenate((w01, coefficients))])


def sample_utility(utility_type: str = 'linear',
                   utility_repr: Optional[np.ndarray] = None) -> UtilityFunction:
    """
    Samples a UtilityFunction of the required utility_type (or constructs one if a utility_repr is provided).
    
    Returns a 3-tuple with the utility's representation for the agent, for the environment and a partial version 
            of the utility function 
    """
    if utility_type == 'linear':
        if utility_repr is not None:
            return LinearUtility(weights=utility_repr)
        weights = sample_linear_weights()
        return LinearUtility(weights=weights)
    elif utility_type == 'threshold':
        if utility_repr is not None:
            return ThresholdUtility(thresholds_and_ceofficients=utility_repr)
        thresholds_and_coefficients = sample_thresholds_and_coefficients()
        return ThresholdUtility(thresholds_and_ceofficients=thresholds_and_coefficients)

    raise ValueError(
        'Expected argument "utility_type" to be a string representing a valid implemented utility')
